"""Command-line interface for incremental Lazada collection runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time
import tomllib
from typing import Any, Dict

from .collector import CollectorConfig, LazadaCollector
from .dom import (
    DEFAULT_SEED_ROOT,
    DomSettings,
    SeleniumDomCollector,
    load_completed_dom_products,
    load_dom_page_progress,
    load_seed_products,
)
from .errors import BlockedError, CollectorError, RateLimitedError
from .history import load_collection_history
from .plan import AutomaticPlan, load_plan
from .quality import attach_quality, evaluate_review
from .scale import (
    ScaleSettings,
    build_scale_summary,
    iter_query_pages,
)
from .schema import ProductRecord
from .storage import CrawlRun
from .transport import extract_item_id
from .transport import load_lazada_cookies


DEFAULT_CONFIG = Path("configs/collector.toml")
DEFAULT_PLAN = Path("configs/collection_plan.toml")


def _read_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Collector config not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _build_collector_config(
    raw: Dict[str, Any],
    args: argparse.Namespace,
) -> CollectorConfig:
    values = dict(raw.get("collector") or {})
    if getattr(args, "transport", None):
        values["transport"] = args.transport
    if getattr(args, "headed", False):
        values["headless"] = False
    profile_dir = getattr(args, "profile_dir", None) or values.get("profile_dir")
    values["profile_dir"] = Path(profile_dir) if profile_dir else None
    cookie_file = getattr(args, "cookie_file", None) or values.get("cookie_file")
    values["cookie_file"] = Path(cookie_file) if cookie_file else None
    return CollectorConfig(**values)


def _output_root(raw: Dict[str, Any], args: argparse.Namespace) -> Path:
    override = getattr(args, "output_root", None)
    configured = (raw.get("output") or {}).get("root", "data/raw")
    return Path(override or configured)


def _parameters(args: argparse.Namespace, config: CollectorConfig) -> Dict[str, Any]:
    values = vars(args).copy()
    values.pop("handler", None)
    values.pop("cookie_file", None)
    for key, value in tuple(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    values["effective_collector"] = {
        "transport": config.transport,
        "page_size": config.page_size,
        "max_pages": config.max_pages,
        "min_delay_seconds": config.min_delay_seconds,
        "max_delay_seconds": config.max_delay_seconds,
        "headless": config.headless,
        "profile_dir": str(config.profile_dir) if config.profile_dir else None,
        "cookie_file_configured": config.cookie_file is not None,
        "review_browser_fallback": config.review_browser_fallback,
    }
    return values


def _plan_summary(plan: AutomaticPlan) -> Dict[str, Any]:
    return {
        "segments": len(plan.segments),
        "queries": plan.query_count,
        "products_per_query": plan.products_per_query,
        "max_products_total": plan.max_products_total,
        "reviews_per_product": plan.reviews_per_product,
        "maximum_accepted_reviews": plan.maximum_accepted_reviews,
        "min_product_reviews": plan.min_product_reviews,
        "scan_multiplier": plan.scan_multiplier,
        "include_sponsored": plan.include_sponsored,
        "rating_filter": plan.rating_filter,
        "quality": {
            "min_chars": plan.quality.min_chars,
            "min_words": plan.quality.min_words,
            "min_unique_word_ratio": plan.quality.min_unique_word_ratio,
            "min_meaningful_words": plan.quality.min_meaningful_words,
            "min_score": plan.quality.min_score,
            "require_vietnamese": plan.quality.require_vietnamese,
            "min_vietnamese_signals": plan.quality.min_vietnamese_signals,
            "max_foreign_script_ratio": plan.quality.max_foreign_script_ratio,
            "reject_suspect_encoding": plan.quality.reject_suspect_encoding,
        },
    }


def _print_result(run: CrawlRun) -> None:
    result = {
        "crawl_id": run.crawl_id,
        "status": run.manifest["status"],
        "counts": run.manifest["counts"],
        "run_dir": str(run.run_dir.resolve()),
        "manifest": str(run.manifest_path.resolve()),
    }
    if "scale" in run.manifest:
        result["scale"] = run.manifest["scale"]
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


def _search(args: argparse.Namespace, raw: Dict[str, Any]) -> int:
    config = _build_collector_config(raw, args)
    run = CrawlRun(
        _output_root(raw, args),
        command="search",
        parameters=_parameters(args, config),
    )
    collector = LazadaCollector(config)
    exit_code = 0
    try:
        for product in collector.search(args.query, limit=args.limit):
            run.add_product(product)
    except CollectorError as exc:
        run.add_error(args.query, str(exc), exc.code)
        run.close("failed")
        exit_code = 2
    except Exception as exc:
        run.add_error(args.query, str(exc), type(exc).__name__)
        run.close("failed")
        exit_code = 2
    finally:
        collector.close()
        run.close()
    _print_result(run)
    return exit_code


def _crawl_product(args: argparse.Namespace, raw: Dict[str, Any]) -> int:
    config = _build_collector_config(raw, args)
    item_id = args.product_id or extract_item_id(args.url)
    if not item_id:
        print("Cannot extract product ID; pass --product-id explicitly.", file=sys.stderr)
        return 2
    product = ProductRecord(
        product_id=item_id,
        name=args.name or f"Product {item_id}",
        url=args.url,
        query=args.query or "",
        seller_id=args.seller_id or "",
    )
    run = CrawlRun(
        _output_root(raw, args),
        command="crawl-product",
        parameters=_parameters(args, config),
    )
    collector = LazadaCollector(config)
    exit_code = 0
    run.add_product(product)
    try:
        for review in collector.iter_reviews(
            product,
            crawl_id=run.crawl_id,
            max_reviews=args.max_reviews,
            rating_filter=args.rating_filter,
        ):
            run.add_review(review)
    except CollectorError as exc:
        run.add_error(item_id, str(exc), exc.code)
        run.close("failed")
        exit_code = 2
    except Exception as exc:
        run.add_error(item_id, str(exc), type(exc).__name__)
        run.close("failed")
        exit_code = 2
    finally:
        collector.close()
        run.close()
    _print_result(run)
    return exit_code


def _crawl_query(args: argparse.Namespace, raw: Dict[str, Any]) -> int:
    config = _build_collector_config(raw, args)
    run = CrawlRun(
        _output_root(raw, args),
        command="crawl-query",
        parameters=_parameters(args, config),
    )
    collector = LazadaCollector(config)
    try:
        products = collector.search(args.query, limit=args.products)
        if not products:
            run.add_error(args.query, "Search returned no products", "NO_PRODUCTS")
            run.close("failed")
            return_code = 2
        else:
            return_code = 0
        for product in products:
            run.add_product(product)
            try:
                for review in collector.iter_reviews(
                    product,
                    crawl_id=run.crawl_id,
                    max_reviews=args.reviews_per_product,
                    rating_filter=args.rating_filter,
                ):
                    run.add_review(review)
            except CollectorError as exc:
                run.add_error(product.product_id, str(exc), exc.code)
            except Exception as exc:
                run.add_error(product.product_id, str(exc), type(exc).__name__)
    except CollectorError as exc:
        run.add_error(args.query, str(exc), exc.code)
        run.close("failed")
        return_code = 2
    except Exception as exc:
        run.add_error(args.query, str(exc), type(exc).__name__)
        run.close("failed")
        return_code = 2
    finally:
        collector.close()
        run.close()
    _print_result(run)
    return return_code


def _crawl_auto(args: argparse.Namespace, raw: Dict[str, Any]) -> int:
    config = _build_collector_config(raw, args)
    plan = load_plan(args.plan)
    if args.max_products is not None:
        plan = replace(plan, max_products_total=args.max_products)
    if args.reviews_per_product is not None:
        plan = replace(plan, reviews_per_product=args.reviews_per_product)
    if args.rating_filter is not None:
        plan = replace(plan, rating_filter=args.rating_filter)
    plan.validate()

    if args.dry_run:
        print(json.dumps(_plan_summary(plan), ensure_ascii=False, indent=2))
        return 0

    output_root = _output_root(raw, args)
    history = load_collection_history(output_root)
    parameters = _parameters(args, config)
    parameters["effective_plan"] = _plan_summary(plan)
    parameters["collection_history"] = {
        "files_scanned": history.files_scanned,
        "cached_products": len(history.cached_products),
        "attempted_products": len(history.attempted_product_ids),
        "accepted_review_ids": len(history.accepted_review_ids),
        "accepted_text_keys": len(history.accepted_text_keys),
        "excluded_review_records": history.excluded_review_records,
    }
    run = CrawlRun(
        output_root,
        command="crawl-auto",
        parameters=parameters,
        existing_review_ids=history.accepted_review_ids,
        existing_text_keys=history.accepted_text_keys,
    )
    collector = LazadaCollector(config)
    randomizer = random.Random(plan.seed)
    selected = []
    selected_ids = set()
    discovery = {
        "queries_attempted": 0,
        "queries_failed": 0,
        "products_seen": 0,
        "products_rejected_sponsored": 0,
        "products_rejected_low_review_count": 0,
        "products_rejected_duplicate": 0,
        "products_rejected_previously_attempted": 0,
        "products_selected": 0,
        "products_selected_from_cache": 0,
        "cache_products_available": len(history.cached_products),
        "search_circuit_opened": False,
    }
    return_code = 0

    try:
        consecutive_discovery_failures = 0
        search_circuit_opened = False
        for segment in plan.segments:
            for query in segment.queries:
                if len(selected) >= plan.max_products_total:
                    break
                discovery["queries_attempted"] += 1
                try:
                    products = collector.search(query, limit=40)
                except CollectorError as exc:
                    discovery["queries_failed"] += 1
                    run.add_error(query, str(exc), exc.code)
                    consecutive_discovery_failures += 1
                    if consecutive_discovery_failures >= 2:
                        discovery["search_circuit_opened"] = True
                        search_circuit_opened = True
                        break
                    continue
                consecutive_discovery_failures = 0
                discovery["products_seen"] += len(products)
                eligible = []
                for product in products:
                    if product.product_id in selected_ids:
                        discovery["products_rejected_duplicate"] += 1
                        continue
                    if (
                        product.product_id in history.attempted_product_ids
                        and not args.allow_previous_products
                    ):
                        discovery["products_rejected_previously_attempted"] += 1
                        continue
                    if product.is_sponsored and not plan.include_sponsored:
                        discovery["products_rejected_sponsored"] += 1
                        continue
                    if (product.review_count or 0) < plan.min_product_reviews:
                        discovery["products_rejected_low_review_count"] += 1
                        continue
                    eligible.append(replace(product, category=segment.name))
                randomizer.shuffle(eligible)
                remaining = plan.max_products_total - len(selected)
                for product in eligible[: min(plan.products_per_query, remaining)]:
                    selected.append(product)
                    selected_ids.add(product.product_id)
                if len(selected) < plan.max_products_total:
                    time.sleep(config.min_delay_seconds)
            if len(selected) >= plan.max_products_total or search_circuit_opened:
                break

        if len(selected) < plan.max_products_total:
            cached_candidates = []
            for product in history.cached_products:
                if product.product_id in selected_ids:
                    continue
                if (
                    product.product_id in history.attempted_product_ids
                    and not args.allow_previous_products
                ):
                    continue
                if product.is_sponsored and not plan.include_sponsored:
                    continue
                if (product.review_count or 0) < plan.min_product_reviews:
                    continue
                cached_candidates.append(product)
            randomizer.shuffle(cached_candidates)
            remaining = plan.max_products_total - len(selected)
            for product in cached_candidates[:remaining]:
                selected.append(product)
                selected_ids.add(product.product_id)
                discovery["products_selected_from_cache"] += 1

        discovery["products_selected"] = len(selected)
        run.manifest["discovery"] = discovery
        if not selected:
            run.add_error(
                "automatic_discovery",
                "No eligible products were discovered",
                "NO_PRODUCTS",
            )
            run.close("failed")
            return_code = 2
        else:
            for product_index, product in enumerate(selected):
                run.add_product(product)

            for product in selected:
                accepted = 0
                max_candidates = plan.reviews_per_product * plan.scan_multiplier
                try:
                    for review in collector.iter_reviews(
                        product,
                        crawl_id=run.crawl_id,
                        max_reviews=max_candidates,
                        rating_filter=plan.rating_filter,
                        max_pages=1,
                    ):
                        run.record_candidate()
                        quality = evaluate_review(review.review_text, plan.quality)
                        if not quality.accepted:
                            run.add_rejection(review, quality)
                            continue
                        enriched = attach_quality(review, quality)
                        if run.add_review(enriched):
                            accepted += 1
                        if accepted >= plan.reviews_per_product:
                            break
                except CollectorError as exc:
                    run.add_error(product.product_id, str(exc), exc.code)
                except Exception as exc:
                    run.add_error(product.product_id, str(exc), type(exc).__name__)
                run.record_product_result(
                    product,
                    accepted_reviews=accepted,
                    target_reviews=plan.reviews_per_product,
                )
                if product_index + 1 < len(selected):
                    time.sleep(
                        randomizer.uniform(
                            config.min_delay_seconds,
                            config.max_delay_seconds,
                        )
                    )
    except Exception as exc:
        run.add_error("crawl-auto", str(exc), type(exc).__name__)
        run.close("failed")
        return_code = 2
    finally:
        collector.close()
        if not run.manifest["finished_at"]:
            if run.manifest["counts"]["products_shortfall"]:
                run.close("completed_with_shortfall")
            else:
                run.close()
    _print_result(run)
    return return_code


def _collect_substantive_product(
    collector: LazadaCollector,
    run: CrawlRun,
    product: ProductRecord,
    plan: AutomaticPlan,
    accepted_target: int,
    candidate_limit: int,
    max_pages: int | None = None,
    quota_shortfall: bool = True,
) -> tuple[int, str | None]:
    """Collect one product and return (accepted reviews, failure code)."""
    accepted = 0
    failure_code = None
    run.add_product(product)
    try:
        for review in collector.iter_reviews(
            product,
            crawl_id=run.crawl_id,
            max_reviews=candidate_limit,
            rating_filter=plan.rating_filter,
            max_pages=max_pages,
        ):
            run.record_candidate()
            quality = evaluate_review(review.review_text, plan.quality)
            if not quality.accepted:
                run.add_rejection(review, quality)
                continue
            if run.add_review(attach_quality(review, quality)):
                accepted += 1
            if accepted >= accepted_target:
                break
    except CollectorError as exc:
        run.add_error(product.product_id, str(exc), exc.code)
        failure_code = exc.code
    except Exception as exc:
        run.add_error(product.product_id, str(exc), type(exc).__name__)
        failure_code = type(exc).__name__
    run.record_product_result(
        product,
        accepted_reviews=accepted,
        target_reviews=(
            accepted_target
            if quota_shortfall or failure_code is not None
            else accepted
        ),
    )
    return accepted, failure_code


def _crawl_scale(args: argparse.Namespace, raw: Dict[str, Any]) -> int:
    config = _build_collector_config(raw, args)
    active_cookie_count = 0
    if config.cookie_file is not None:
        active_cookie_count = len(load_lazada_cookies(config.cookie_file))
    plan = load_plan(args.plan)
    if args.reviews_per_product is not None:
        plan = replace(plan, reviews_per_product=args.reviews_per_product)
    if args.rating_filter is not None:
        plan = replace(plan, rating_filter=args.rating_filter)
    plan.validate()
    settings = ScaleSettings(
        target_reviews=args.target_reviews,
        reviews_per_product=plan.reviews_per_product,
        max_search_pages=args.max_search_pages,
        products_per_page=args.products_per_page,
        cooldown_seconds=args.cooldown_seconds,
        max_cooldown_seconds=args.max_cooldown_seconds,
        max_cooldowns=args.max_cooldowns,
        failure_threshold=args.failure_threshold,
        max_products=args.max_products,
    )
    settings.validate()

    output_root = _output_root(raw, args)
    history = load_collection_history(output_root)
    summary = build_scale_summary(
        plan,
        settings,
        existing_reviews=len(history.accepted_review_ids),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "crawl-scale",
                    **summary,
                    "history": {
                        "quality_records_counted": len(
                            history.accepted_review_ids
                        ),
                        "older_or_invalid_records_excluded": (
                            history.excluded_review_records
                        ),
                    },
                    "cookie_session": {
                        "configured": config.cookie_file is not None,
                        "active_lazada_cookies": active_cookie_count,
                    },
                    "plan": _plan_summary(plan),
                    "checkpoint": "after_each_product",
                    "resume": "rerun_the_same_command",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if summary["reviews_remaining"] == 0:
        print(
            json.dumps(
                {
                    "status": "already_satisfied",
                    **summary,
                    "run_created": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parameters = _parameters(args, config)
    parameters["effective_plan"] = _plan_summary(plan)
    parameters["scale_estimate"] = summary
    parameters["cookie_session"] = {
        "configured": config.cookie_file is not None,
        "active_lazada_cookies": active_cookie_count,
        "values_persisted": False,
    }
    parameters["collection_history"] = {
        "files_scanned": history.files_scanned,
        "cached_products": len(history.cached_products),
        "attempted_products": len(history.attempted_product_ids),
        "accepted_review_ids": len(history.accepted_review_ids),
        "accepted_text_keys": len(history.accepted_text_keys),
        "excluded_review_records": history.excluded_review_records,
    }
    run = CrawlRun(
        output_root,
        command="crawl-scale",
        parameters=parameters,
        existing_review_ids=history.accepted_review_ids,
        existing_text_keys=history.accepted_text_keys,
    )
    collector = LazadaCollector(config)
    randomizer = random.Random(plan.seed)
    existing_reviews = len(history.accepted_review_ids)
    processed_ids: set[str] = set()
    consecutive_search_failures = 0
    consecutive_review_failures = 0
    stop_status: str | None = None
    return_code = 0
    discovery = {
        "cache_products_available": len(history.cached_products),
        "products_selected_from_cache": 0,
        "products_selected_from_search": 0,
        "products_seen": 0,
        "products_rejected_duplicate": 0,
        "products_rejected_previously_attempted": 0,
        "products_rejected_sponsored": 0,
        "products_rejected_low_review_count": 0,
        "search_pages_attempted": 0,
        "search_pages_failed": 0,
    }
    run.manifest["discovery"] = discovery
    run.manifest["scale"] = {
        "target_reviews_total": settings.target_reviews,
        "reviews_before_run": existing_reviews,
        "reviews_total": existing_reviews,
        "reviews_remaining": settings.target_reviews - existing_reviews,
        "products_processed": 0,
        "cooldowns_used": 0,
        "consecutive_cooldowns": 0,
        "last_cooldown_reason": None,
        "last_cooldown_seconds": 0,
    }

    def target_reached() -> bool:
        return (
            existing_reviews + run.manifest["counts"]["reviews_written"]
            >= settings.target_reviews
        )

    def product_limit_reached() -> bool:
        return (
            settings.max_products is not None
            and run.manifest["scale"]["products_processed"]
            >= settings.max_products
        )

    def update_progress() -> None:
        total = existing_reviews + run.manifest["counts"]["reviews_written"]
        run.manifest["scale"]["reviews_total"] = total
        run.manifest["scale"]["reviews_remaining"] = max(
            0, settings.target_reviews - total
        )

    def checkpoint_and_report(product: ProductRecord, accepted: int) -> None:
        update_progress()
        run.checkpoint()
        scale = run.manifest["scale"]
        print(
            (
                f"[progress] total={scale['reviews_total']}/"
                f"{settings.target_reviews} new={run.manifest['counts']['reviews_written']} "
                f"products={scale['products_processed']} "
                f"last_product={product.product_id} accepted={accepted} "
                f"errors={run.manifest['counts']['errors']}"
            ),
            file=sys.stderr,
            flush=True,
        )

    def cooldown(reason: str) -> bool:
        consecutive = run.manifest["scale"]["consecutive_cooldowns"]
        if consecutive >= settings.max_cooldowns:
            return False
        number = consecutive + 1
        delay = settings.cooldown_delay(number)
        run.manifest["scale"]["cooldowns_used"] += 1
        run.manifest["scale"]["consecutive_cooldowns"] = number
        run.manifest["scale"]["last_cooldown_reason"] = reason
        run.manifest["scale"]["last_cooldown_seconds"] = delay
        run.checkpoint()
        print(
            (
                f"[cooldown] reason={reason} attempt={number}/"
                f"{settings.max_cooldowns} wait={delay:.0f}s"
            ),
            file=sys.stderr,
            flush=True,
        )
        if delay:
            time.sleep(delay)
        collector.reset_transports()
        return True

    def eligible(product: ProductRecord) -> bool:
        if product.product_id in processed_ids:
            discovery["products_rejected_duplicate"] += 1
            return False
        if (
            product.product_id in history.attempted_product_ids
            and not args.allow_previous_products
        ):
            discovery["products_rejected_previously_attempted"] += 1
            return False
        if product.is_sponsored and not plan.include_sponsored:
            discovery["products_rejected_sponsored"] += 1
            return False
        if (product.review_count or 0) < plan.min_product_reviews:
            discovery["products_rejected_low_review_count"] += 1
            return False
        return True

    def process_product(product: ProductRecord, source: str) -> bool:
        nonlocal consecutive_review_failures, stop_status
        processed_ids.add(product.product_id)
        if source == "cache":
            discovery["products_selected_from_cache"] += 1
        else:
            discovery["products_selected_from_search"] += 1
        remaining = (
            settings.target_reviews
            - existing_reviews
            - run.manifest["counts"]["reviews_written"]
        )
        accepted_target = min(settings.reviews_per_product, remaining)
        accepted, failure_code = _collect_substantive_product(
            collector,
            run,
            product,
            plan,
            accepted_target=accepted_target,
            candidate_limit=(
                settings.reviews_per_product * plan.scan_multiplier
            ),
            max_pages=args.review_pages_per_product,
            quota_shortfall=False,
        )
        run.manifest["scale"]["products_processed"] += 1
        checkpoint_and_report(product, accepted)

        if failure_code == "BLOCKED":
            if not cooldown("review_endpoint_blocked"):
                stop_status = "paused_rate_limit"
                return False
            consecutive_review_failures = 0
        elif failure_code is not None:
            consecutive_review_failures += 1
            if consecutive_review_failures >= settings.failure_threshold:
                if not cooldown("review_transport_failures"):
                    stop_status = "paused_rate_limit"
                    return False
                consecutive_review_failures = 0
        else:
            consecutive_review_failures = 0
            run.manifest["scale"]["consecutive_cooldowns"] = 0
        if target_reached() or product_limit_reached():
            return False
        time.sleep(
            randomizer.uniform(
                config.min_delay_seconds,
                config.max_delay_seconds,
            )
        )
        return True

    try:
        cached_candidates = [
            product
            for product in history.cached_products
            if eligible(product)
        ]
        randomizer.shuffle(cached_candidates)
        for product in cached_candidates:
            if target_reached() or product_limit_reached() or stop_status:
                break
            if not process_product(product, "cache"):
                break

        if not target_reached() and not product_limit_reached() and not stop_status:
            for segment, query, page in iter_query_pages(plan, settings):
                if target_reached() or product_limit_reached() or stop_status:
                    break
                discovery["search_pages_attempted"] += 1
                try:
                    products = collector.search(query, limit=40, page=page)
                except CollectorError as exc:
                    discovery["search_pages_failed"] += 1
                    run.add_error(f"{query}:page={page}", str(exc), exc.code)
                    consecutive_search_failures += 1
                    if consecutive_search_failures >= settings.failure_threshold:
                        if not cooldown("catalogue_search_failures"):
                            stop_status = "paused_rate_limit"
                            break
                        consecutive_search_failures = 0
                    continue
                except Exception as exc:
                    discovery["search_pages_failed"] += 1
                    run.add_error(
                        f"{query}:page={page}",
                        str(exc),
                        type(exc).__name__,
                    )
                    consecutive_search_failures += 1
                    if consecutive_search_failures >= settings.failure_threshold:
                        if not cooldown("catalogue_search_failures"):
                            stop_status = "paused_rate_limit"
                            break
                        consecutive_search_failures = 0
                    continue

                consecutive_search_failures = 0
                run.manifest["scale"]["consecutive_cooldowns"] = 0
                discovery["products_seen"] += len(products)
                page_candidates = []
                for product in products:
                    if eligible(product):
                        page_candidates.append(
                            replace(product, category=segment.name)
                        )
                randomizer.shuffle(page_candidates)
                for product in page_candidates[: settings.products_per_page]:
                    if target_reached() or product_limit_reached() or stop_status:
                        break
                    if not process_product(product, "search"):
                        break
                if not target_reached() and not product_limit_reached() and not stop_status:
                    time.sleep(config.min_delay_seconds)

        if target_reached():
            stop_status = "completed_target"
        elif stop_status is None and product_limit_reached():
            stop_status = "stopped_max_products"
        elif stop_status is None:
            stop_status = "exhausted_catalogue"
            return_code = 3
        elif stop_status == "paused_rate_limit":
            return_code = 3
    except KeyboardInterrupt:
        stop_status = "interrupted"
        return_code = 130
        print(
            "[interrupted] Checkpoint saved; rerun the same command to continue.",
            file=sys.stderr,
        )
    except Exception as exc:
        run.add_error("crawl-scale", str(exc), type(exc).__name__)
        stop_status = "failed"
        return_code = 2
    finally:
        update_progress()
        run.checkpoint()
        collector.close()
        run.close(stop_status or "failed")

    _print_result(run)
    return return_code


def _crawl_dom_scale(args: argparse.Namespace, raw: Dict[str, Any]) -> int:
    """Build a large corpus from visible review cards in a local Chrome."""
    config = _build_collector_config(raw, args)
    dom_cookie_session = {
        "configured": config.cookie_file is not None,
        "valid_export": None,
        "active_lazada_cookies": 0,
        "values_written_to_dataset": False,
        "browser_profile_may_persist": config.profile_dir is not None,
    }
    if config.cookie_file is not None:
        try:
            dom_cookie_session["active_lazada_cookies"] = len(
                load_lazada_cookies(config.cookie_file)
            )
            dom_cookie_session["valid_export"] = True
        except CollectorError:
            # A persistent Chrome profile can remain usable even after the
            # standalone API cookie export expires.
            dom_cookie_session["valid_export"] = False
    plan = load_plan(args.plan)
    plan.validate()
    settings = DomSettings(
        max_pages_per_product=args.max_pages_per_product,
        page_delay_min=args.page_delay_min,
        page_delay_max=args.page_delay_max,
        load_timeout_seconds=args.load_timeout_seconds,
    )
    settings.validate()

    output_root = _output_root(raw, args)
    history = load_collection_history(output_root)
    completed_before = load_completed_dom_products(output_root)
    page_progress_before = load_dom_page_progress(output_root)
    seeds = load_seed_products(args.seed_root)
    existing_reviews = len(history.accepted_review_ids)
    reviews_remaining = max(0, args.target_reviews - existing_reviews)

    cached_by_id = {
        product.product_id: product
        for product in history.cached_products
    }
    seed_by_id = {product.product_id: product for product in seeds}
    static_by_id = {**seed_by_id, **cached_by_id}
    static_products = sorted(
        static_by_id.values(),
        key=lambda product: (
            -(product.review_count or 0),
            product.product_id,
        ),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "crawl-dom-scale",
                    "cost": "self_hosted_no_paid_scraping_service",
                    "target_reviews_total": args.target_reviews,
                    "reviews_already_collected": existing_reviews,
                    "reviews_remaining": reviews_remaining,
                    "cached_products": len(cached_by_id),
                    "repository_seed_products": len(seed_by_id),
                    "unique_static_products": len(static_by_id),
                    "completed_dom_products": len(completed_before),
                    "products_with_page_checkpoints": len(page_progress_before),
                    "automatic_search_queries": plan.query_count,
                    "maximum_automatic_search_pages": (
                        plan.query_count * args.max_search_pages
                    ),
                    "max_pages_per_product": settings.max_pages_per_product,
                    "cookie_session": dom_cookie_session,
                    "quality": _plan_summary(plan)["quality"],
                    "checkpoint": "after_each_rendered_review_page",
                    "resume": "rerun_the_same_command",
                    "privacy": "buyer_names_avatars_and_cookies_are_not_written",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if reviews_remaining == 0:
        print(
            json.dumps(
                {
                    "status": "already_satisfied",
                    "target_reviews_total": args.target_reviews,
                    "reviews_total": existing_reviews,
                    "run_created": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parameters = _parameters(args, config)
    parameters["effective_dom"] = {
        "max_pages_per_product": settings.max_pages_per_product,
        "page_delay_min": settings.page_delay_min,
        "page_delay_max": settings.page_delay_max,
        "product_delay_min": args.product_delay_min,
        "product_delay_max": args.product_delay_max,
        "load_timeout_seconds": settings.load_timeout_seconds,
        "uses_public_review_json_endpoint": False,
        "captcha_bypass": False,
    }
    parameters["cookie_session"] = dom_cookie_session
    parameters["effective_plan"] = _plan_summary(plan)
    parameters["collection_history"] = {
        "files_scanned": history.files_scanned,
        "accepted_review_ids": existing_reviews,
        "accepted_text_keys": len(history.accepted_text_keys),
        "completed_dom_products": len(completed_before),
        "products_with_page_checkpoints": len(page_progress_before),
        "repository_seed_products": len(seed_by_id),
        "cached_products": len(cached_by_id),
    }
    run = CrawlRun(
        output_root,
        command="crawl-dom-scale",
        parameters=parameters,
        existing_review_ids=history.accepted_review_ids,
        existing_text_keys=history.accepted_text_keys,
    )
    run.manifest["dom"] = {
        "target_reviews_total": args.target_reviews,
        "reviews_before_run": existing_reviews,
        "reviews_total": existing_reviews,
        "reviews_remaining": reviews_remaining,
        "products_processed": 0,
        "pages_processed": 0,
        "products_from_cache_or_seeds": 0,
        "products_from_rendered_search": 0,
        "search_pages_processed": 0,
        "consecutive_review_suppressions": 0,
        "completed_product_ids": [],
        "page_progress": {},
        "current_product_id": None,
        "current_page": None,
        "transport": "selenium_dom",
    }
    run.checkpoint()

    randomizer = random.Random(plan.seed)
    processed_ids: set[str] = set()
    stop_status: str | None = None
    return_code = 0
    collector: SeleniumDomCollector | None = None
    consecutive_suppressions = 0

    def target_reached() -> bool:
        return (
            existing_reviews + run.manifest["counts"]["reviews_written"]
            >= args.target_reviews
        )

    def product_limit_reached() -> bool:
        return (
            args.max_products is not None
            and run.manifest["dom"]["products_processed"] >= args.max_products
        )

    def update_totals() -> None:
        total = existing_reviews + run.manifest["counts"]["reviews_written"]
        run.manifest["dom"]["reviews_total"] = total
        run.manifest["dom"]["reviews_remaining"] = max(
            0,
            args.target_reviews - total,
        )

    def report_page(
        product: ProductRecord,
        page_number: int,
        total_pages: int | None,
        page_accepted: int,
    ) -> None:
        update_totals()
        run.checkpoint()
        total_label = str(total_pages) if total_pages is not None else "?"
        print(
            (
                f"[dom-progress] total={run.manifest['dom']['reviews_total']}/"
                f"{args.target_reviews} "
                f"new={run.manifest['counts']['reviews_written']} "
                f"products={run.manifest['dom']['products_processed'] + 1} "
                f"product={product.product_id} "
                f"page={page_number}/{total_label} "
                f"accepted_page={page_accepted} "
                f"errors={run.manifest['counts']['errors']}"
            ),
            file=sys.stderr,
            flush=True,
        )

    def process_product(product: ProductRecord, source: str) -> bool:
        nonlocal consecutive_suppressions, stop_status, return_code
        if collector is None:
            raise RuntimeError("DOM collector is not initialized")
        if product.product_id in processed_ids:
            return True
        if (
            product.product_id in completed_before
            and not args.allow_completed_products
        ):
            return True
        processed_ids.add(product.product_id)
        run.add_product(product)
        run.manifest["dom"]["current_product_id"] = product.product_id
        run.manifest["dom"]["current_page"] = None
        accepted_product = 0
        iteration_finished = False
        failure_code = None
        start_page = page_progress_before.get(product.product_id, 0) + 1
        try:
            if start_page > settings.max_pages_per_product:
                iteration_finished = True
            else:
                if start_page > 1:
                    print(
                        (
                            f"[dom-resume] product={product.product_id} "
                            f"start_page={start_page}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                for dom_page in collector.iter_review_pages(
                    product,
                    crawl_id=run.crawl_id,
                    start_page=start_page,
                ):
                    if dom_page.reviews:
                        consecutive_suppressions = 0
                        run.manifest["dom"][
                            "consecutive_review_suppressions"
                        ] = 0
                    page_accepted = 0
                    for review in dom_page.reviews:
                        run.record_candidate()
                        quality = evaluate_review(review.review_text, plan.quality)
                        if not quality.accepted:
                            run.add_rejection(review, quality)
                            continue
                        if run.add_review(attach_quality(review, quality)):
                            accepted_product += 1
                            page_accepted += 1
                    run.manifest["dom"]["pages_processed"] += 1
                    run.manifest["dom"]["current_page"] = dom_page.page_number
                    run.manifest["dom"]["page_progress"][
                        product.product_id
                    ] = dom_page.page_number
                    report_page(
                        product,
                        dom_page.page_number,
                        dom_page.total_pages,
                        page_accepted,
                    )
                    if target_reached():
                        break
                else:
                    iteration_finished = True
        except RateLimitedError as exc:
            run.add_error(product.product_id, str(exc), exc.code)
            failure_code = exc.code
            consecutive_suppressions += 1
            run.manifest["dom"][
                "consecutive_review_suppressions"
            ] = consecutive_suppressions
            if consecutive_suppressions >= 3:
                stop_status = "paused_rate_limit"
                return_code = 3
        except BlockedError as exc:
            run.add_error(product.product_id, str(exc), exc.code)
            failure_code = exc.code
            stop_status = "paused_challenge"
            return_code = 3
        except CollectorError as exc:
            run.add_error(product.product_id, str(exc), exc.code)
            failure_code = exc.code
        except Exception as exc:
            run.add_error(
                product.product_id,
                str(exc),
                type(exc).__name__,
            )
            failure_code = type(exc).__name__

        # Only a normally exhausted paginator or a confirmed unavailable
        # product is complete. Transport/schema failures remain resumable.
        sampling_complete = (
            iteration_finished and failure_code is None
        ) or (
            failure_code == "PRODUCT_UNAVAILABLE"
        )
        if sampling_complete:
            run.manifest["dom"]["completed_product_ids"].append(
                product.product_id
            )
        run.record_product_result(
            product,
            accepted_reviews=accepted_product,
            target_reviews=accepted_product,
        )
        run.manifest["dom"]["products_processed"] += 1
        if source == "rendered_search":
            run.manifest["dom"]["products_from_rendered_search"] += 1
        else:
            run.manifest["dom"]["products_from_cache_or_seeds"] += 1
        update_totals()
        run.checkpoint()
        if stop_status or target_reached() or product_limit_reached():
            return False
        delay = randomizer.uniform(
            args.product_delay_min,
            args.product_delay_max,
        )
        if delay:
            time.sleep(delay)
        return True

    try:
        collector = SeleniumDomCollector(
            headless=config.headless,
            profile_dir=config.profile_dir,
            cookie_file=config.cookie_file,
            settings=settings,
            seed=plan.seed,
        )
        run.manifest["dom"]["cookie_session"] = {
            **dom_cookie_session,
            "cookies_imported_into_chrome": (
                collector.transport.cookies_imported
            ),
            "import_failed": collector.transport.cookie_import_failed,
        }
        run.checkpoint()
        for product in static_products:
            if target_reached() or product_limit_reached() or stop_status:
                break
            if not process_product(product, "static"):
                break

        if not target_reached() and not product_limit_reached() and not stop_status:
            search_settings = ScaleSettings(
                max_search_pages=args.max_search_pages,
                products_per_page=args.products_per_search,
            )
            for segment, query, page in iter_query_pages(plan, search_settings):
                if target_reached() or product_limit_reached() or stop_status:
                    break
                try:
                    products = collector.search_products(
                        query,
                        page=page,
                        category=segment.name,
                        limit=max(args.products_per_search * 2, 20),
                    )
                except BlockedError as exc:
                    run.add_error(
                        f"{query}:page={page}",
                        str(exc),
                        exc.code,
                    )
                    stop_status = "paused_challenge"
                    return_code = 3
                    break
                except CollectorError as exc:
                    run.add_error(
                        f"{query}:page={page}",
                        str(exc),
                        exc.code,
                    )
                    continue
                run.manifest["dom"]["search_pages_processed"] += 1
                randomizer.shuffle(products)
                selected = [
                    product
                    for product in products
                    if product.product_id not in processed_ids
                    and (
                        args.allow_completed_products
                        or product.product_id not in completed_before
                    )
                ][: args.products_per_search]
                for product in selected:
                    if target_reached() or product_limit_reached() or stop_status:
                        break
                    if not process_product(product, "rendered_search"):
                        break
                run.checkpoint()
                if (
                    not target_reached()
                    and not product_limit_reached()
                    and not stop_status
                    and args.search_delay_seconds
                ):
                    time.sleep(args.search_delay_seconds)

        if target_reached():
            stop_status = "completed_target"
        elif stop_status is None and product_limit_reached():
            stop_status = "stopped_max_products"
        elif stop_status is None:
            stop_status = "exhausted_catalogue"
            return_code = 3
    except KeyboardInterrupt:
        stop_status = "interrupted"
        return_code = 130
        print(
            "[interrupted] DOM checkpoint saved; rerun the same command to continue.",
            file=sys.stderr,
            flush=True,
        )
    except CollectorError as exc:
        run.add_error("crawl-dom-scale", str(exc), exc.code)
        stop_status = "failed"
        return_code = 2
    except Exception as exc:
        run.add_error("crawl-dom-scale", str(exc), type(exc).__name__)
        stop_status = "failed"
        return_code = 2
    finally:
        update_totals()
        run.manifest["dom"]["current_product_id"] = None
        run.manifest["dom"]["current_page"] = None
        run.checkpoint()
        if collector is not None:
            collector.close()
        run.close(stop_status or "failed")

    _print_result(run)
    return return_code


def _add_common_crawl_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transport",
        choices=("auto", "requests", "selenium"),
        default=None,
        help="Override the transport configured in collector.toml.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Chrome when Selenium is used.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Optional local Chrome profile directory; never commit it.",
    )
    parser.add_argument(
        "--cookie-file",
        default=None,
        help=(
            "Netscape cookie export. Only active lazada.vn cookies are "
            "loaded; cookie values are never written to output."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the configured output root.",
    )
    parser.add_argument(
        "--rating-filter",
        type=int,
        choices=range(0, 6),
        default=0,
        metavar="0..5",
        help="0 preserves the natural distribution; 1..5 is enriched sampling.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lazada-collect",
        description="Collect a versioned, provenance-aware Lazada review corpus.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML configuration path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Save a product search snapshot.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--output-root", default=None)
    search.set_defaults(handler=_search)

    product = subparsers.add_parser(
        "crawl-product",
        help="Collect reviews for one product.",
    )
    product.add_argument("url")
    product.add_argument("--product-id", default=None)
    product.add_argument("--name", default=None)
    product.add_argument("--seller-id", default=None)
    product.add_argument("--query", default=None)
    product.add_argument("--max-reviews", type=int, default=100)
    _add_common_crawl_options(product)
    product.set_defaults(handler=_crawl_product)

    query = subparsers.add_parser(
        "crawl-query",
        help="Search products and collect reviews incrementally.",
    )
    query.add_argument("query")
    query.add_argument("--products", type=int, default=5)
    query.add_argument("--reviews-per-product", type=int, default=50)
    _add_common_crawl_options(query)
    query.set_defaults(handler=_crawl_query)

    automatic = subparsers.add_parser(
        "crawl-auto",
        help="Discover products across configured segments and retain substantive reviews.",
    )
    automatic.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        help="Automatic sampling and quality plan.",
    )
    automatic.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Override the plan's total product cap.",
    )
    automatic.add_argument(
        "--reviews-per-product",
        type=int,
        default=None,
        help="Override the accepted-review target per product.",
    )
    automatic.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without network access.",
    )
    automatic.add_argument(
        "--allow-previous-products",
        action="store_true",
        help="Allow products already attempted in earlier runs.",
    )
    _add_common_crawl_options(automatic)
    automatic.set_defaults(handler=_crawl_auto, rating_filter=None)

    scale = subparsers.add_parser(
        "crawl-scale",
        help="Build or resume a 30k-50k substantive-review corpus.",
    )
    scale.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        help="Automatic sampling and quality plan.",
    )
    scale.add_argument(
        "--target-reviews",
        type=int,
        default=30_000,
        help="Target total unique accepted reviews under the output root.",
    )
    scale.add_argument(
        "--reviews-per-product",
        type=int,
        default=None,
        help="Accepted-review quota per product (default: plan value).",
    )
    scale.add_argument(
        "--max-search-pages",
        type=int,
        default=40,
        help="Maximum catalogue pages per configured query.",
    )
    scale.add_argument(
        "--products-per-page",
        type=int,
        default=15,
        help="Maximum eligible products sampled from each search page.",
    )
    scale.add_argument(
        "--review-pages-per-product",
        type=int,
        default=5,
        help=(
            "Maximum natural-distribution API review pages scanned per "
            "product (50 candidates per page by default)."
        ),
    )
    scale.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Optional per-run product cap, useful for a pilot.",
    )
    scale.add_argument(
        "--cooldown-seconds",
        type=float,
        default=300.0,
        help="Initial wait after a repeated transport failure.",
    )
    scale.add_argument(
        "--max-cooldown-seconds",
        type=float,
        default=900.0,
        help="Maximum exponential-backoff wait.",
    )
    scale.add_argument(
        "--max-cooldowns",
        type=int,
        default=3,
        help="Maximum consecutive cooldown cycles before pausing safely.",
    )
    scale.add_argument(
        "--failure-threshold",
        type=int,
        default=3,
        help="Consecutive failures before a cooldown.",
    )
    scale.add_argument(
        "--allow-previous-products",
        action="store_true",
        help="Allow products already attempted in earlier runs.",
    )
    scale.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate capacity from local history without network access.",
    )
    _add_common_crawl_options(scale)
    scale.set_defaults(handler=_crawl_scale, rating_filter=None)

    dom_scale = subparsers.add_parser(
        "crawl-dom-scale",
        help=(
            "Collect substantive reviews from rendered product pages in a "
            "self-hosted Chrome session."
        ),
    )
    dom_scale.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        help="Automatic sampling and quality plan.",
    )
    dom_scale.add_argument(
        "--target-reviews",
        type=int,
        default=20_000,
        help="Target total unique accepted reviews under data/raw.",
    )
    dom_scale.add_argument(
        "--seed-root",
        type=Path,
        default=DEFAULT_SEED_ROOT,
        help="Directory containing optional product URL seed text files.",
    )
    dom_scale.add_argument(
        "--max-pages-per-product",
        type=int,
        default=100,
        help="Maximum rendered review pages visited for one product.",
    )
    dom_scale.add_argument(
        "--max-search-pages",
        type=int,
        default=40,
        help="Maximum rendered catalogue pages per configured query.",
    )
    dom_scale.add_argument(
        "--products-per-search",
        type=int,
        default=12,
        help="Maximum products selected from each rendered search page.",
    )
    dom_scale.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Optional product cap for a pilot run.",
    )
    dom_scale.add_argument(
        "--page-delay-min",
        type=float,
        default=2.0,
        help="Minimum delay between rendered review pages.",
    )
    dom_scale.add_argument(
        "--page-delay-max",
        type=float,
        default=4.0,
        help="Maximum delay between rendered review pages.",
    )
    dom_scale.add_argument(
        "--product-delay-min",
        type=float,
        default=4.0,
        help="Minimum delay between products.",
    )
    dom_scale.add_argument(
        "--product-delay-max",
        type=float,
        default=8.0,
        help="Maximum delay between products.",
    )
    dom_scale.add_argument(
        "--search-delay-seconds",
        type=float,
        default=5.0,
        help="Delay after each rendered catalogue page.",
    )
    dom_scale.add_argument(
        "--load-timeout-seconds",
        type=float,
        default=35.0,
        help="Maximum wait for a product/search page to render.",
    )
    dom_scale.add_argument(
        "--allow-completed-products",
        action="store_true",
        help="Revisit products fully exhausted by earlier DOM runs.",
    )
    dom_scale.add_argument(
        "--headed",
        action="store_true",
        help="Show the dedicated Chrome window.",
    )
    dom_scale.add_argument(
        "--profile-dir",
        default=None,
        help="Optional dedicated local Chrome profile directory.",
    )
    dom_scale.add_argument(
        "--cookie-file",
        default=None,
        help=(
            "Optional Netscape cookie export. Only active lazada.vn cookies "
            "are imported into Chrome, and values are never persisted."
        ),
    )
    dom_scale.add_argument(
        "--output-root",
        default=None,
        help="Override the configured output root.",
    )
    dom_scale.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect local capacity and settings without opening Chrome.",
    )
    dom_scale.set_defaults(handler=_crawl_dom_scale)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "limit") and args.limit < 1:
        parser.error("--limit must be positive")
    if hasattr(args, "products") and args.products < 1:
        parser.error("--products must be positive")
    if hasattr(args, "max_reviews") and args.max_reviews < 1:
        parser.error("--max-reviews must be positive")
    if (
        hasattr(args, "reviews_per_product")
        and args.reviews_per_product is not None
        and args.reviews_per_product < 1
    ):
        parser.error("--reviews-per-product must be positive")
    if hasattr(args, "max_products") and args.max_products is not None:
        if args.max_products < 1:
            parser.error("--max-products must be positive")
    for name in (
        "target_reviews",
        "max_search_pages",
        "products_per_page",
        "products_per_search",
        "max_pages_per_product",
        "review_pages_per_product",
        "failure_threshold",
    ):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "max_cooldowns") and args.max_cooldowns < 0:
        parser.error("--max-cooldowns cannot be negative")
    if hasattr(args, "cooldown_seconds") and args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    if (
        hasattr(args, "max_cooldown_seconds")
        and args.max_cooldown_seconds < args.cooldown_seconds
    ):
        parser.error(
            "--max-cooldown-seconds must be >= --cooldown-seconds"
        )
    for minimum_name in (
        "page_delay_min",
        "product_delay_min",
        "search_delay_seconds",
    ):
        if hasattr(args, minimum_name) and getattr(args, minimum_name) < 0:
            parser.error(f"--{minimum_name.replace('_', '-')} cannot be negative")
    for minimum_name, maximum_name in (
        ("page_delay_min", "page_delay_max"),
        ("product_delay_min", "product_delay_max"),
    ):
        if (
            hasattr(args, minimum_name)
            and getattr(args, maximum_name) < getattr(args, minimum_name)
        ):
            parser.error(
                f"--{maximum_name.replace('_', '-')} must be >= "
                f"--{minimum_name.replace('_', '-')}"
            )
    if (
        hasattr(args, "load_timeout_seconds")
        and args.load_timeout_seconds <= 0
    ):
        parser.error("--load-timeout-seconds must be positive")
    try:
        raw = _read_config(args.config)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))
    return int(args.handler(args, raw))


if __name__ == "__main__":
    raise SystemExit(main())
