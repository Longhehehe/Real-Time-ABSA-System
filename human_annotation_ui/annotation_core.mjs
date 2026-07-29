export const WORKSPACE_SCHEMA_VERSION = "human-absa-workspace/2.0.0";
export const ASSIGNMENT_SCHEMA_VERSION = "human-absa-assignment/1.0.0";
export const EXPORT_SCHEMA_VERSION = "human-absa-export/2.0.0";
export const UI_VERSION = "human-absa-ui/2.0.0";

export const ASPECTS = Object.freeze([
  "Chất lượng sản phẩm",
  "Hiệu năng & Trải nghiệm",
  "Đúng mô tả",
  "Giá cả & Khuyến mãi",
  "Vận chuyển",
  "Đóng gói",
  "Dịch vụ & Thái độ Shop",
  "Bảo hành & Đổi trả",
  "Tính xác thực",
]);

export const UNCERTAINTY_CODES = Object.freeze([
  "ASPECT_BOUNDARY",
  "POLARITY_SCOPE",
  "SARCASM",
  "INSUFFICIENT_CONTEXT",
  "TYPO_LANGUAGE",
  "NON_REVIEW",
  "BOILERPLATE_MIXED_WITH_REVIEW",
  "OTHER",
]);

export const ASPECT_UNCERTAINTY_CODES = Object.freeze([
  "ASPECT_BOUNDARY",
  "POLARITY_SCOPE",
  "SARCASM",
  "INSUFFICIENT_CONTEXT",
  "TYPO_LANGUAGE",
  "OTHER",
]);

export const CANONICAL_LABELS = Object.freeze([2, -1, 0, 1, "1, -1"]);

export function canonicalJson(value) {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  const keys = Object.keys(value).sort();
  return `{${keys
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
}

export function codePointLength(value) {
  return Array.from(value).length;
}

export function codePointSlice(value, start, end) {
  return Array.from(value).slice(start, end).join("");
}

export function occurrenceAtOffset(text, quote, start) {
  if (!quote || codePointSlice(text, start, start + codePointLength(quote)) !== quote) {
    return 0;
  }
  const textPoints = Array.from(text);
  const quotePoints = Array.from(quote);
  let occurrence = 0;
  for (let index = 0; index <= start; index += 1) {
    let matches = true;
    for (let offset = 0; offset < quotePoints.length; offset += 1) {
      if (textPoints[index + offset] !== quotePoints[offset]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      occurrence += 1;
      if (index === start) {
        return occurrence;
      }
    }
  }
  return 0;
}

export function newRecordState(annotationId, reviewTextSha256) {
  return {
    annotation_id: annotationId,
    review_text_sha256: reviewTextSha256,
    annotation_status: "",
    aspects: ASPECTS.map((aspect) => ({
      aspect,
      label: null,
      evidence: [],
      uncertainty_codes: [],
    })),
    review_uncertainty_codes: [],
    notes: "",
    read_complete: false,
    revisit: false,
    complete: false,
    completed_at: null,
    updated_at: null,
    revision_number: 0,
    revisions: [],
  };
}

function hasExactKeys(value, expected) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length &&
    actual.every((key, index) => key === wanted[index]);
}

export function validateRecord(reviewText, record, annotatorId = "") {
  const errors = [];
  const add = (message) => errors.push(message);
  const allowedStatuses = ["LABELED", "ESCALATE", "REJECT_NON_REVIEW"];
  const status = record?.annotation_status;
  if (!allowedStatuses.includes(status)) {
    add("Chưa chọn trạng thái review.");
  }
  if (!record?.read_complete) {
    add("Chưa xác nhận đã đọc toàn bộ review.");
  }
  if (!String(annotatorId || "").trim()) {
    add("Chưa nhập mã người gán.");
  }
  if (!Array.isArray(record?.aspects) || record.aspects.length !== ASPECTS.length) {
    add("Phải có đúng chín aspect theo thứ tự canonical.");
    return { valid: false, errors, normalized: null };
  }

  const reviewCodes = Array.isArray(record.review_uncertainty_codes)
    ? record.review_uncertainty_codes
    : [];
  if (new Set(reviewCodes).size !== reviewCodes.length) {
    add("Mã không chắc chắn cấp review bị lặp.");
  }
  for (const code of reviewCodes) {
    if (!UNCERTAINTY_CODES.includes(code)) {
      add(`Mã không chắc chắn cấp review không hợp lệ: ${String(code)}.`);
    }
  }

  const normalizedAspects = [];
  const allCodes = [...reviewCodes];
  record.aspects.forEach((row, index) => {
    const expectedAspect = ASPECTS[index];
    const prefix = `Aspect ${index + 1} · ${expectedAspect}`;
    if (!hasExactKeys(row, ["aspect", "label", "evidence", "uncertainty_codes"])) {
      add(`${prefix}: cấu trúc dữ liệu không hợp lệ.`);
      return;
    }
    if (row.aspect !== expectedAspect) {
      add(`${prefix}: sai tên hoặc thứ tự aspect.`);
    }
    const codes = Array.isArray(row.uncertainty_codes)
      ? row.uncertainty_codes
      : [];
    if (new Set(codes).size !== codes.length) {
      add(`${prefix}: mã không chắc chắn bị lặp.`);
    }
    for (const code of codes) {
      if (!UNCERTAINTY_CODES.includes(code)) {
        add(`${prefix}: mã không chắc chắn không hợp lệ ${String(code)}.`);
      }
    }
    allCodes.push(...codes);

    const evidence = Array.isArray(row.evidence) ? row.evidence : [];
    const normalizedEvidence = [];
    const spans = new Set();
    for (const item of evidence) {
      if (
        !hasExactKeys(item, ["quote", "start", "end", "polarity"]) ||
        !Number.isInteger(item.start) ||
        !Number.isInteger(item.end) ||
        item.start < 0 ||
        item.end <= item.start ||
        !["positive", "negative", "neutral"].includes(item.polarity)
      ) {
        add(`${prefix}: evidence có cấu trúc/offset/polarity không hợp lệ.`);
        continue;
      }
      const quote = codePointSlice(reviewText, item.start, item.end);
      if (!item.quote || quote !== item.quote) {
        add(`${prefix}: evidence không còn khớp nguyên văn review.`);
        continue;
      }
      const spanKey = `${item.start}:${item.end}`;
      if (spans.has(spanKey)) {
        add(`${prefix}: cùng một evidence occurrence bị lặp.`);
        continue;
      }
      spans.add(spanKey);
      const occurrence = occurrenceAtOffset(
        reviewText,
        item.quote,
        item.start,
      );
      if (occurrence < 1) {
        add(`${prefix}: không xác định được occurrence của evidence.`);
        continue;
      }
      normalizedEvidence.push({
        quote: item.quote,
        start: item.start,
        end: item.end,
        occurrence,
        polarity: item.polarity,
      });
    }

    const label = row.label;
    if (status === "REJECT_NON_REVIEW") {
      if (label !== null || normalizedEvidence.length > 0) {
        add(`${prefix}: mẫu non-review phải có label null và không evidence.`);
      }
    } else if (!CANONICAL_LABELS.some((candidate) => candidate === label)) {
      add(`${prefix}: chưa chọn một trong năm nhãn canonical.`);
    } else {
      const polarities = new Set(
        normalizedEvidence.map((item) => item.polarity),
      );
      if (label === 2 && normalizedEvidence.length > 0) {
        add(`${prefix}: nhãn 2 không được có evidence.`);
      }
      if (label !== 2 && normalizedEvidence.length === 0) {
        add(`${prefix}: nhãn mentioned phải có exact evidence.`);
      }
      if (
        label === 1 &&
        (!polarities.has("positive") || polarities.has("negative"))
      ) {
        add(`${prefix}: nhãn 1 cần positive và không có negative evidence.`);
      }
      if (
        label === -1 &&
        (!polarities.has("negative") || polarities.has("positive"))
      ) {
        add(`${prefix}: nhãn -1 cần negative và không có positive evidence.`);
      }
      if (
        label === 0 &&
        (polarities.size !== 1 || !polarities.has("neutral"))
      ) {
        add(`${prefix}: nhãn 0 chỉ dùng neutral evidence.`);
      }
      if (
        label === "1, -1" &&
        (!polarities.has("positive") || !polarities.has("negative"))
      ) {
        add(`${prefix}: mixed cần cả positive và negative evidence.`);
      }
    }
    normalizedAspects.push({
      aspect: expectedAspect,
      label,
      evidence: normalizedEvidence,
      uncertainty_codes: [...codes],
    });
  });

  const notes = typeof record?.notes === "string" ? record.notes : "";
  const hasUncertainty = allCodes.length > 0;
  if (status === "REJECT_NON_REVIEW") {
    if (!reviewCodes.includes("NON_REVIEW")) {
      add("Không phải review phải có mã NON_REVIEW ở cấp review.");
    }
    if (!notes.trim()) {
      add("Không phải review phải có lý do ngắn trong ghi chú.");
    }
  } else if (allCodes.includes("NON_REVIEW")) {
    add("Mã NON_REVIEW chỉ hợp lệ với trạng thái Không phải review.");
  }
  if (
    allCodes.includes("BOILERPLATE_MIXED_WITH_REVIEW") &&
    status !== "ESCALATE"
  ) {
    add("BOILERPLATE_MIXED_WITH_REVIEW bắt buộc trạng thái Cần hòa giải.");
  }
  if (allCodes.includes("OTHER") && !notes.trim()) {
    add("Mã OTHER bắt buộc có giải thích trong ghi chú.");
  }
  if (status === "LABELED" && hasUncertainty) {
    add("Có uncertainty code thì trạng thái phải là Cần hòa giải.");
  }
  if (status === "ESCALATE" && !hasUncertainty) {
    add("Cần hòa giải phải có ít nhất một uncertainty code.");
  }

  return {
    valid: errors.length === 0,
    errors,
    normalized: errors.length === 0
      ? {
          annotation_status: status,
          aspects: normalizedAspects,
          review_uncertainty_codes: [...reviewCodes],
          notes,
        }
      : null,
  };
}

export function isRecordUncertain(record) {
  return (
    record.annotation_status === "ESCALATE" ||
    record.review_uncertainty_codes.length > 0 ||
    record.aspects.some((row) => row.uncertainty_codes.length > 0)
  );
}
