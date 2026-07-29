---
title: "Quy trình xây dựng bộ dữ liệu Lazada tiếng Việt cho bài toán ABSA đa cực"
subtitle: "Living protocol, nhật ký thực nghiệm và hồ sơ truy xuất dữ liệu"
author: "Real-Time ABSA System"
date: "2026-07-25"
lang: vi-VN
---

# Kiểm soát tài liệu

Tài liệu này là hồ sơ sống của toàn bộ quá trình xây dựng bộ dữ liệu. Mỗi tác
vụ kỹ thuật phải được ghi lại ngay sau khi thực hiện và bản DOCX phải được sinh
lại từ bản nguồn Markdown. Những nội dung ghi là “kế hoạch” không được trình
bày trong paper như một bước đã thực nghiệm. Chỉ các mục có trạng thái
“đã thực hiện và xác thực” mới được dùng làm bằng chứng thực nghiệm.

| Trường | Giá trị hiện tại |
|---|---|
| Bản tài liệu | 0.9-v2.1.2-audited-preannotation |
| Ngày khởi tạo | 2026-07-25 |
| Bản nguồn | `docs/DATASET_CONSTRUCTION_PROTOCOL.md` |
| Bản phân phối | `docs/DATASET_CONSTRUCTION_PROTOCOL.docx` |
| Raw snapshot | `data/raw/` |
| Parent release bất biến | `lazada_vi_reviews_v1_20260725` |
| Snapshot curation đã thay thế | `lazada_vi_absa_curation_v2_20260725`; `lazada_vi_absa_curation_v2_1_20260725`; `lazada_vi_absa_curation_v2_1_1_20260725` |
| Curation release hiện hành | `lazada_vi_absa_curation_v2_1_2_20260725` |
| Release ID hiện hành | `lazada-vi-absa-curation-e42d6c5319faedd3` |
| Trạng thái hiện hành | Validator và rebuild-check byte-for-byte đã đạt; semantic holdout agent-QC đã chạy; pre-annotation, chưa phải gold/final |
| Bài toán đích | Aspect-Based Sentiment Analysis đa khía cạnh, đa cực |

# Mục tiêu nghiên cứu và đơn vị dữ liệu

Mục tiêu là xây dựng một corpus đánh giá thương mại điện tử tiếng Việt có
nguồn gốc truy xuất được, phù hợp để huấn luyện và đánh giá mô hình ABSA. Một
đơn vị dữ liệu là nội dung của một đánh giá gắn với sản phẩm, thời điểm,
rating, nguồn thu thập và metadata kỹ thuật. Danh tính người mua, avatar,
cookie, email và số điện thoại không được lưu như structured release fields;
raw/verbatim text vẫn có residual PII risk và local cookie/profile secret tồn
tại ngoài release.

Bài toán sử dụng chín aspect category:

1. Chất lượng sản phẩm.
2. Hiệu năng & Trải nghiệm.
3. Đúng mô tả.
4. Giá cả & Khuyến mãi.
5. Vận chuyển.
6. Đóng gói.
7. Dịch vụ & Thái độ Shop.
8. Bảo hành & Đổi trả.
9. Tính xác thực.

Mỗi aspect nhận một trong các nhãn `-1`, `0`, `1`, `2`, hoặc chuỗi đa cực
`1, -1`. Trong đó `-1` là tiêu cực, `0` là đề cập trung tính, `1` là tích cực,
`2` là không được đề cập, và `1, -1` biểu thị cùng một aspect chứa cả bằng
chứng tích cực lẫn tiêu cực. Ô trống chỉ là trạng thái “chưa gán nhãn”, không
được đồng nhất với nhãn `2`.

# Nguyên tắc thiết kế bắt buộc

## Bất biến dữ liệu gốc

`data/raw/` là tầng append-only. Làm sạch không được sửa nội dung, thay thế
review, hoặc xóa run lịch sử. Mọi quyết định loại/gộp phải xuất hiện trong một
ledger dẫn ngược tới `crawl_id`, file nguồn và số dòng nguồn.

## Tách biệt các tầng dữ liệu

- **Raw layer:** dữ liệu đúng như collector đã ghi.
- **Canonical layer:** các record hợp lệ về schema và provenance, chưa loại
  vì nội dung đáng ngờ.
- **Clean-core layer:** các review đạt tiêu chí tự động có độ chính xác cao.
- **Quarantine layer:** template, off-topic hoặc duplicate còn cần duyệt.
- **Annotated layer:** chỉ được tạo sau khi hoàn tất gán nhãn và adjudication.
- **Model split layer:** train/dev/test chỉ được chốt sau khi có nhãn; toàn bộ
  product và duplicate/template family phải nằm trong cùng một split.

## Không đánh đổi chất lượng lấy số lượng

Mục tiêu 30.000 hay 50.000 review không phải tiêu chí khoa học để giữ một dòng
dữ liệu. Một release nhỏ hơn nhưng có tiêu chí nhất quán và kiểm định thủ công
có giá trị hơn một release lớn chứa spam, câu mẫu và leakage.

## Quyết định có thể đảo ngược

Không xóa vật lý record vì heuristic. Mỗi record phải có `decision`,
`reason_codes`, `representative_id`, `cluster_id`, phiên bản rule và bằng
chứng định lượng. Một ngưỡng thay đổi phải có thể tái tạo release mới từ raw.

# Dữ liệu nguồn lịch sử

Hai snapshot lịch sử được giữ riêng để audit/so sánh; chúng chưa phải hai
benchmark độc lập:

- Old Dataset: 10.105 review không rỗng, 9.773 nội dung duy nhất.
- Augmented Dataset: 11.202 review không rỗng, 10.877 nội dung duy nhất.

Các số trên dùng exact Python string: không cộng cơ học hai nguồn thành 21.307
mẫu. Có 9.945 dòng augmented exact-match một dòng old; exact-unique
intersection là 9.620 và union chỉ 11.030 nội dung. Trong augmented,
`aug_strategy` rỗng ở 9.945 dòng và `llm_synthetic` ở 1.257 dòng. Nếu chuẩn
hóa NFKC+casefold+collapse-whitespace, Old/Aug/union unique lần lượt là
9.772/10.876/11.029; vì vậy paper phải nêu rõ định nghĩa “unique”.

Augmented phần lớn là snapshot dẫn xuất chứa lại Old, không phải domain hoặc
test benchmark độc lập. Hai file không được dùng như hai test set rồi so sánh
như bằng chứng lặp độc lập; synthetic chỉ được bổ sung vào **train split** sau
khi split dữ liệu thật và phải có real-only vs real+synthetic ablation. Tính
tương thích nhãn giữa hai snapshot chưa được audit có hệ thống theo từng cặp
review--aspect, nên chưa đưa trực tiếp vào corpus mới. Nhãn đa cực `1, -1` là
một nhãn hợp lệ, không được gọi là “xung đột nhãn”. Chỉ báo annotation conflict
khi cùng review/aspect có hai nhãn không tương thích sau chuẩn hóa schema.

Các XLSX copy nằm dưới `legacy/data/`. Audit read-only, schema, định nghĩa
exact/normalized key và toàn bộ file hash đã được persist tại
`docs/audits/LEGACY_DATASET_AUDIT.md` và
`docs/audits/legacy_xlsx_sha256.csv`; SHA-256 của inventory CSV là
`fd7cfcd7ec7fc139017ee265f113482e71ad0c15520ef1570de708670d18396a`.
Tuy nhiên inventory này nằm ngoài frozen curation release, chưa có
label-compatibility validator và worktree chưa có commit/tag; cần phát hành
một benchmark release riêng trước khi dùng làm gold evidence trong paper.

# Giai đoạn thu thập dữ liệu mới — đã thực hiện

## Kế hoạch lấy mẫu

Collector tự động tìm sản phẩm không quảng cáo trong tám phân đoạn:
electronics, home appliances, beauty and personal care, home living, fashion,
mother and baby, sports and outdoors, food and beverage. Rating filter bằng
0 để không áp quota cứng theo sao. Vì đây là dữ liệu quan sát từ các sản phẩm
được truy cập thành công và các review vượt bộ lọc, corpus không được tuyên bố
là đại diện xác suất cho toàn bộ Lazada.

## Hai transport

1. Cookie-authenticated review API lấy `model.items[*].reviewContent` và
   `reviewRateId`.
2. Selenium DOM đọc nội dung trong card thuộc
   `#module_product_review`.

Hai transport được chạy bằng các command riêng. Config đóng băng có
`review_browser_fallback=false`, vì vậy 197 run `crawl-dom-scale` là fallback
vận hành được khởi động rõ ràng sau các run API, không phải API tự chuyển sang
Selenium trong cùng một run. Collector không bypass CAPTCHA. Giá trị cookie
không được ghi vào review/release; manifest có ghi metadata/count của
`cookie_session`. Local cookie export và browser profile có thể còn tồn tại
ngoài release và phải được coi là secret.

## Schema và provenance

Review record chứa `crawl_id`, `collected_at`, `collector_version`,
`collection_transport`, `sampling_frame`, `query`, `product_id`, `seller_id`,
`source_url`, `review_id`, `review_text`, `rating`, `review_time`, `sku_info`,
`page_number`, `response_sha256`, category và các quality metrics. Tất cả
review của policy hiện hành dùng schema 1.2 và sampling frame `natural`.

## Bộ lọc substantive_vi_v2 ban đầu

Điều kiện ban đầu:

- ít nhất 80 ký tự;
- ít nhất 15 từ;
- unique-word ratio ít nhất 0,40;
- ít nhất 8 từ có nghĩa;
- quality score ít nhất 0,55;
- có ít nhất 2 tín hiệu tiếng Việt;
- foreign-script ratio không quá 0,20;
- loại encoding có dấu hiệu mojibake.

Bộ lọc này kiểm tra tính dài và hình thức ngôn ngữ. Nó **không** chứng minh
nội dung là review thật, có liên quan tới sản phẩm, hoặc không phải template.

## Kết quả thu thập và đóng băng raw

- 393/393 manifest đã đóng, có 393 `crawl_id` khác nhau; không còn manifest
  ở trạng thái `running`.
- Cửa sổ thu thập theo UTC từ
  `2026-07-23T12:34:20.849523Z` đến
  `2026-07-25T07:15:14.247830Z`, tương ứng 19:34 ngày 23/07 đến 14:15 ngày
  25/07 theo UTC+07, tổng thời gian tường 42 giờ 40 phút 53 giây.
- Trạng thái run gồm 1 `completed`, 7 `completed_target`, 3
  `completed_with_shortfall`, 108 `stopped_max_products`, 197
  `paused_rate_limit`, 72 `failed` và 5 `interrupted_external`. “Đã đóng”
  nghĩa là manifest có trạng thái kết thúc và file có thể kiểm toán; không có
  nghĩa mọi run đều thu đủ quota.
- Lệnh đã dùng gồm 5 `crawl-auto`, 191 `crawl-scale` và 197
  `crawl-dom-scale`. Các run API và DOM dùng chung lịch sử review ID/text để
  chống trùng qua nhiều phiên.
- Tổng trên manifest là 69.744 candidate, phân rã khép kín thành 35.996 dòng
  bị bộ lọc chất lượng từ chối, 1.799 dòng deduplicate và 31.949 dòng được
  ghi. Phương trình kiểm tra là
  \(69.744 = 35.996 + 1.799 + 31.949\).
- 31.949 dòng review JSON hợp lệ khớp tổng `reviews_written`; 31.928 review
  thuộc `substantive_vi_v2`, còn 21 dòng thuộc policy cũ
  `substantive_v1`. Parent release chỉ nhận 31.928 dòng của policy hiện hành.
- 1.304 lượt xử lý product trong manifest không phải 1.304 sản phẩm độc lập:
  file product có 983 product ID duy nhất, raw review được chấp nhận có 977,
  parent có 974 và clean-core v2.1.2 còn 971.
- Có 24 query tự nhiên được cấu hình trên tám phân đoạn cùng bốn seed label;
  chúng tạo 28 label ở file product nhưng chỉ 27 query label trong raw review
  vì seed `seed:url7_n` không có review được chấp nhận. Không được mô tả đây
  là 28 truy vấn tìm kiếm độc lập.
- Hai cặp exact duplicate trong toàn bộ 31.949 dòng đều thuộc 21 dòng policy
  cũ. Tập `substantive_vi_v2` không trùng exact review ID hoặc exact
  normalized text trước khi xử lý alias chéo transport/punctuation.
- Source inventory có 1.573 đường dẫn đã hash, gồm 1.572 crawl artifact và
  `data/raw/.gitkeep`; không có bằng chứng response page bị replay để tăng số
  dòng.

Phân bố rating của canonical v1 rất lệch: 30.386 review 5 sao (95,17%), 657
review 4 sao, 282 review 3 sao, 151 review 2 sao và 452 review 1 sao. Đây là
một hạn chế phải được báo cáo và xử lý bằng metric, sampling hoặc loss ở
train; không được sửa rating hay sinh dữ liệu trước khi split.

Đây là **purposive accepted-review frame**, không phải probability sample của
Lazada. Bộ lọc ban đầu loại 35.996/69.744 candidate (51,61%), chủ yếu vì review
ngắn hoặc không đạt ngưỡng ngôn ngữ/chất lượng. Cửa sổ thu thập ngắn, lựa chọn
24 query, sản phẩm truy cập thành công, rate limit, transport fallback và độ
lệch 5 sao đều tạo selection bias. Cách diễn đạt được phép trong paper là
“phân bố tự nhiên trong khung review dài được chọn có chủ đích”; không được
tuyên bố corpus đại diện cho toàn bộ Lazada hoặc toàn bộ người mua Việt Nam.

Trên 35.996 rejection row, multi-label reason counts là: 34.025
`TOO_SHORT_CHARS`, 26.924 `TOO_FEW_WORDS`, 22.324 `LOW_QUALITY_SCORE`, 17.420
`TOO_FEW_MEANINGFUL_WORDS`, 4.767 `INSUFFICIENT_VIETNAMESE_SIGNAL`, 572
`LOW_LEXICAL_DIVERSITY`, 127 `FOREIGN_SCRIPT_DOMINANT` và 40
`SUSPECT_ENCODING`. Các reason overlap và không được cộng để suy tổng; chúng
xác nhận selection chủ yếu ưu tiên review dài/có đủ từ.

# Release kỹ thuật v1 và lý do không sử dụng — đã thực hiện

Release `lazada_vi_reviews_v1_20260725` xác nhận được 31.928 record duy nhất
theo exact review ID và nội dung sau NFKC/casefold/whitespace. Release này đã
chia 31.797 dòng vào primary annotation và 131 dòng vào manual review. Tuy
nhiên, audit ngữ nghĩa sau đó chứng minh ngưỡng full-text 5-gram Jaccard 0,85
quá bảo thủ và bộ lọc không tách clause theo dấu phẩy.

Các phát hiện khiến v1 bị dừng trước annotation:

- ít nhất 49 cặp API--DOM là cùng một review nhưng khác ID, định dạng ngày,
  ngôn ngữ của field heading hoặc punctuation;
- 1.560 review lặp substantive clause bên trong;
- 5.841 review có ít nhất 50% nội dung là clause xuất hiện từ 10 lần;
- 3.013 review hoàn toàn được ghép từ các clause lặp;
- 17 nội dung Gboard, quảng cáo livestream, tin tức, cảnh báo và các văn bản
  không liên quan xuất hiện trong trường review;
- 598 dòng có disclosure “nhận/kiếm xu”; không phải tất cả đều vô dụng;
- screening bảo thủ tạo 892 dòng cần quarantine ngay.

Các con số trên dùng định nghĩa khác nhau và có overlap; không được cộng lại
để suy ra tổng số dòng loại. Kết luận đúng là: v1 sạch về cấu trúc nhưng chưa
sạch về ngữ nghĩa.

# Quy trình xây dựng clean release v2.1.2 — đã thực hiện và xác thực kỹ thuật

Các bước dưới đây đã được cài đặt và chạy trên toàn bộ 31.928 record của parent
release. Snapshot rule 2.0.0, 2.1.0 và 2.1.1 được giữ làm mốc lịch sử nhưng đã
bị thay thế sau semantic audit và QC. Release hiện hành dùng rule 2.1.2 và
schema curation 2.1. Từ “xác thực” trong phần này chỉ có nghĩa là schema,
checksum, partition, transformation chain và các invariant kỹ thuật đã qua
validator; nó không có nghĩa là 20.622 dòng clean-core đã được con người duyệt
hết.

## Bước 1 — Khóa input và xác minh provenance

1. Xác minh checksum closure của parent release và 49 checksum entries trước
   khi đọc record.
2. Đối chiếu parent manifest với 1.573 source-inventory entry đã hash
   (1.572 crawl artifact cộng `.gitkeep`) và 393 crawl manifest đã đóng.
3. Đọc đúng 31.928 canonical record thuộc policy `substantive_vi_v2`; fail
   closed nếu sample ID trùng, JSON hỏng, schema sai hoặc checksum thay đổi.
4. Cấm các khóa nhận dạng người mua, cookie, token, email và số điện thoại
   trong record đầu ra.
5. Gắn cho mỗi dòng số thứ tự canonical, đường dẫn file raw, số dòng raw,
   `crawl_id`, transport và hash nội dung để truy ngược.

## Bước 2 — Chuẩn hóa chỉ để so sánh

Giữ nguyên `review_text`. Tạo các view phụ:

- NFKC, casefold và collapse whitespace;
- punctuation-insensitive view;
- token sequence;
- clause sequence, có tách dấu phẩy có điều kiện;
- canonical bilingual headings, ví dụ
  `Effectiveness` ↔ `Hiệu quả`, `Fragrance` ↔ `Hương thơm`;
- ngày review chuẩn ISO và SKU normalization bảo toàn chữ số.

Các view phụ chỉ dùng cho dedup/flag, không thay thế nội dung gốc.

## Bước 3 — Dedup record cùng nguồn

Review có cùng API `reviewRateId` là hard duplicate. Ngoài exact source ID,
pipeline lập khóa nội dung sau NFKC/casefold/whitespace và khóa bỏ qua
punctuation/symbol. Các submission có ID khác nhưng cùng khóa nội dung được
coi là **text duplicate phục vụ chống leakage**, không được tuyên bố là cùng
một danh tính review. Điều này đặc biệt quan trọng vì 21/57 punctuation-alias
nằm ở product khác nhau. Representative được chọn theo identity API, chất
lượng, độ đầy đủ metadata và mức phạt punctuation; mọi alias vẫn được giữ
trong ledger.

## Bước 4 — Dedup chéo API--DOM

Candidate chỉ được tạo khi cùng product, rating, ngày chuẩn hóa và SKU tương
thích. Sau khi canonical hóa bounded bilingual headings, thực hiện one-to-one
matching theo similarity. Giữ API record làm representative vì có numeric
source ID và metadata verified-purchase; DOM ID được lưu làm alias. Block
nhiều-nhiều hoặc thiếu date/SKU phải đi manual, không auto-merge.

Ngưỡng đã chạy là word-5-gram Jaccard tối thiểu 0,50 và length ratio tối thiểu
0,75 trong block metadata trên; ambiguous margin là 0,10. Kết quả có 84
candidate, 83 alias one-to-one được xác nhận và một candidate mơ hồ không bị
gộp. Cộng với 57 punctuation-exact alias, tổng cộng 140 alias được
`EXCLUDE_AUTO`; đây là lý do duy nhất được phép auto-exclude ở release 2.1.2.

## Bước 5 — Hard non-review filters

Các mẫu hệ thống có precision cao như hướng dẫn Gboard, OTP/SMS nhà mạng,
livestream/share URL, lời mời inbox/hotline, tuyển dụng, pasted-news và văn
bản hoàn toàn lạc đề được đưa vào quarantine với reason code cụ thể. Regex
không được dùng một từ đơn như “shop”, “mua ngay” hoặc dấu hỏi để loại vì có
thể xuất hiện trong complaint hợp lệ.

Tất cả match nội dung ở bước này chỉ vào `QUARANTINE`, kể cả signature có độ
chính xác cao; không xóa vật lý và không auto-exclude. Taxonomy đã chạy gồm:
Gboard/share-card, SMS nhà mạng, tin pháp lý, tin tức/mạng xã hội, văn bản
giáo dục, thông báo riêng, tuyển dụng, game/platform text, quảng cáo người
bán, copied content, contact bị làm rối, URL/domain và personal abuse. Rule
quảng cáo cần tổ hợp tín hiệu cụ thể hoặc CTA lặp ở ít nhất hai nội dung khác
nhau của cùng product; từ khóa đơn lẻ không đủ.

## Bước 6 — Template và boilerplate

Review dạng danh sách dấu phẩy được tách clause khi có ít nhất ba segment
ngắn. Tính document frequency theo review và distinct-product frequency.
Template family là đối tượng khác record duplicate:

- duplicate: cùng underlying review, chỉ giữ một representative;
- template: các submission khác ID nhưng dùng phrase inventory giống nhau,
  giữ provenance và quarantine/downweight;
- boilerplate disclosure: phần “hình ảnh nhận xu” có thể đi kèm review hữu
  ích, cần flag hoặc re-score, không xóa mù cả dòng.

Global threshold similarity thấp không được dùng để xóa connected component,
vì transitive chaining có thể nối hàng trăm review khác nhau.

Near-duplicate là tầng candidate riêng, không đồng nhất với 140 duplicate đã
xác nhận. Global candidate yêu cầu word-5-gram Jaccard ≥ 0,92 và length ratio
≥ 0,90; within-product candidate dùng Jaccard ≥ 0,85 và length ratio ≥ 0,80.
Release có 48 cặp, 44 cluster và 47 non-representative candidate. Similarity
đơn lẻ không auto-merge/auto-delete; candidate ambiguous hoặc
non-representative đi review queue.

Template frequency threshold hiện hành là: global candidate có review-DF ≥ 5,
product-DF ≥ 3, ít nhất hai recurrent clause và coverage ≥ 0,40; global-high
có ít nhất ba clause và coverage ≥ 0,70; product-high có review-DF ≥ 10, ít
nhất ba clause và coverage ≥ 0,85. Frequency chỉ tạo evidence/quarantine.

Các snapshot 2.1.0--2.1.1 dùng structural catalogue detector cho danh sách
feature ngắn: đếm số segment, delimiter, tỷ lệ segment ngắn, marketing opener,
recurrent clause coverage, trailing delimiter và buyer-residual clause. Sau
audit 2.1.0, mọi `TEMPLATE_GLOBAL_CANDIDATE` được chuyển sang quarantine và
hai nhánh `RECURRENT_SPEC_LIST`/`LOW_DENSITY_BROCHURE_LIST` được bổ sung.
Con số 953 từng ghi trong calibration là snapshot dự đoán tại thời điểm hiệu
chỉnh, không phải số hit cuối của release và không được dùng thay cho manifest.

Unseen semantic audit của clean-core 2.1.1 chọn 200 dòng mới, loại các ID đã
dùng ở audit/calibration trước. Một analyst/agent đọc nội dung đánh giá 179/200
(89,5%; Wilson 95% 84,48--93,03%) là strict ABSA-usable; còn 16 pure
catalogue, 2 hard non-review và 3 borderline. Vì 21 trường hợp này sau đó được
dùng để phát triển 2.1.2, audit 2.1.1 trở thành **calibration evidence** cho
rule mới, không còn là unbiased estimate của release kế tiếp.

Rule 2.1.2 bổ sung ba nhánh cấu trúc bảo thủ:

- `MODULAR_TITLE_LIST`: ít nhất hai segment kiểu title/list; short ratio ≥
  0,75 nếu có trailing delimiter hoặc ≥ 0,60 nếu text đã clean; capitalized
  initial ratio ≥ 0,60 hoặc clause tái diễn; không có reviewer anchor đáng tin;
- `EXPANDED_RECURRENT_SPEC_LIST`: ít nhất hai clause thông số mở rộng, có
  recurrence, capitalized initial ratio ≥ 0,60, không còn buyer-residual và đủ
  dấu hiệu list/title;
- `GLUED_CATALOGUE_CLAUSE`: phát hiện ranh giới lower-to-upper bị dính, phần
  sau mở bằng catalogue phrase, trong review có ít nhất ba segment và có
  trailing/cleaned evidence.

Reviewer veto giữ dấu tiếng Việt để tránh ánh xạ sai như `tối→toi`,
`túi→tui`, `trà→tra`, `chống→chong`. Veto chỉ chấp nhận cụm thể hiện hành
động/trải nghiệm có ngữ cảnh; không dùng một buyer token rộng để miễn mọi danh
sách catalogue.

Trên toàn bộ parent 2.1.2, ba rule mới có lần lượt 8.033, 1.207 và 145 hit;
union là 8.324 dòng. Trong union này, 6.032 dòng chồng với structural rule cũ
và 2.292 chỉ do nhánh mới. Tổng
`STRUCTURAL_CATALOGUE_NO_EXPERIENCE` tăng từ 7.282 ở 2.1.1 lên 9.575 ở 2.1.2,
không làm mất hit cũ. Release đồng thời có 1.848
`TEMPLATE_GLOBAL_CANDIDATE`, 1.319 `TEMPLATE_PRODUCT_HIGH` và 194 template
family. Các reason có overlap; tất cả chỉ dẫn tới quarantine.

Calibration dương xác định bằng stable hash lấy 120 candidate
`MODULAR_TITLE_LIST`: analyst/agent đánh giá cả 120 là pure hoặc mixed
catalogue, chưa thấy clean-natural false positive; Wilson lower bound khoảng
96,9% chỉ áp dụng cho panel này. Negative control 100 dòng ở sát ranh giới
nhưng được reviewer-veto giữ lại có 9 clean-natural, 33 mixed, 57 pure template
và 1 unrelated: veto bảo vệ cả 42 natural/mixed nhưng cố ý bỏ lọt 58 dòng
không hợp lệ để ưu tiên precision. Đây không phải gold/human panel.
ID/order/label ledger của hai panel 120/100 này không được persist tại thời
điểm calibration, nên các count chỉ là descriptive calibration evidence,
không phải artifact có thể tái tạo độc lập.

Hai mươi mốt target từ audit 2.1.1 đều bị quarantine sau sửa: 16/16 pure
catalogue bởi structural evidence, 2/2 non-review bởi quảng cáo và ba
borderline bởi no-evaluation/glued-catalogue. Kết quả 21/21 chỉ là regression
trên tập đã dùng để hiệu chỉnh, không phải bằng chứng recall tổng quát.

## Bước 7 — Repetition và quality sau decontamination

Phát hiện clause lặp, kể cả clause ngăn bởi dấu phẩy. Xây decontaminated view
chỉ để đánh giá xem phần còn lại có tiếp tục đạt quality hay không. Nội dung
gốc vẫn bất biến. Nếu review chỉ vượt min-length nhờ repetition/template,
record được quarantine.

Các biến đổi deterministic đã chạy theo thứ tự, luôn trên
`curated_review_text`, không sửa `review_text`:

1. NFKC và collapse whitespace.
2. Bỏ đúng pure reward-disclaimer clause khi clause không còn bằng chứng ABSA;
   mixed clause chỉ được flag.
3. Thu gọn exact repeated clause theo span có ranh giới.
4. Thay phone/email/URL bằng placeholder và lưu offset/hash của span bị thay.
5. Phát hiện internal 4-gram lặp với ít nhất ba n-gram lặp và later-token
   coverage tối thiểu 0,15.
6. Phát hiện keyboard-smash bằng strong token hoặc ít nhất hai medium token,
   có corpus document-frequency guard, periodic-string guard và natural
   elongation guard.
7. Chỉ cắt một ASCII lowercase terminal token dài 7--12 ký tự, không có nguyên
   âm sau accent-fold, root document frequency ≤ 2, có ranh giới hợp lệ,
   không trùng query/SKU metadata và phần còn lại vẫn đạt toàn bộ quality
   policy; trường hợp embedded, nhiều token, residual ngắn hoặc mơ hồ chỉ bị
   quarantine nguyên văn.
8. Chạy lại bộ lọc `substantive_vi_v2` trên văn bản sau biến đổi.

Ledger hiện có 2.377 phép biến đổi: 1.910 exact-clause collapse, 413 reward
disclaimer removal, 39 PII redaction và 15 terminal keyboard-smash removal.
Có 193 record biến đổi vẫn đủ điều kiện vào clean-core dưới trạng thái
`KEEP_CLEANED`; các record biến đổi còn lại vẫn ở quarantine nếu có tín hiệu
nghi ngờ khác.

Mười terminal trim mới của 2.1.2 được replay độc lập từ raw: cả 10 source
review ID/product ID/text, span, input/output/token/removed hash, detector
evidence và residual-quality đều khớp. Các suffix bị bỏ là chuỗi ASCII thường
7--12 ký tự không có nguyên âm. Tám record vẫn ở quarantine do nhiễu/template
khác; hai residual review hợp lệ vào `KEEP_CLEANED`. Kết hợp với năm trim đã
audit ở 2.1.1, chưa thấy semantic content bị cắt nhầm trong 15 trường hợp đã
kiểm, nhưng đây là technical/content audit một analyst, không phải annotation
gold.

## Bước 8 — Relevance và manual adjudication

Những record không thuộc hard rule nhưng có off-topic score, foreign/machine
translation, reward disclosure hoặc template coverage trung gian đi vào hàng
đợi manual. Mỗi quyết định phải có ít nhất hai lựa chọn độc lập khi xây gold
corpus: keep, exclude hoặc escalate. Không tự động xóa structured prompt như
`Chất liệu:` và không loại câu hỏi/khiếu nại chỉ vì có dấu `?`.

Reward detector dùng chữ Unicode gốc để không nhầm `xử/sự` với `xu/su`. Guard
bảo vệ các câu trải nghiệm hợp lệ như “không có ảnh minh họa”, “giống hình
minh họa”, “màu đậm hơn so với hình minh họa” và complaint về phụ kiện hiển
thị trong ảnh. Detector keyboard-smash không dùng giá trị sau dấu `:` làm bằng
chứng độc lập, vì audit cho thấy cách đó bắt nhầm nhiều structured ABSA field.
Mọi regex/rule mới chỉ được chấp nhận sau positive và negative regression
tests.

Rule 2.1.2 giữ tổ hợp ngữ cảnh cho câu viết nhận xét để lấy voucher, quảng cáo
“pass lại” kèm giá, danh sách size/sale/đặt trước, direct call-to-action và lời
khuyên mua hàng chung không chứa bằng chứng aspect; đồng thời bổ sung seller
voice, listing-title và hai chuỗi “chưa dùng/chưa đánh giá hiện tại” có guard
để không bắt nhầm service review. Detector
Tagalog-dominant yêu cầu ít nhất 15 token Latin, ít nhất 5 lượt xuất hiện của
ít nhất 4 function word Tagalog khác nhau và tỷ lệ tối thiểu 0,15. Các match
này đều vào quarantine, không auto-delete.

Có 80 exact-ID QC escalation duy nhất từ các vòng đọc nội dung: 27 pure
template, 6 borderline, 45 mixed/noisy và 2 non-review. Cả 80 nằm đúng một lần
trong `QUARANTINE`, có trạng thái `MANUAL_PENDING`, không annotation-eligible
và xuất hiện trong review queue. Chúng chưa phải nhãn người thật và vẫn chờ
adjudication. Exact-ID escalation được công khai trong config để có thể kiểm
toán hoặc đảo ngược, nhưng không thay thế detector tổng quát; các tỷ lệ trên
tập QC đã biết không được trình bày như generalization performance.

## Bước 9 — Product cap và kiểm soát concentration

Release canonical hiện hành **không áp product cap**. Pipeline chỉ ghi
deterministic within-product rank để audit; không loại một review hợp lệ chỉ
vì product có nhiều dòng. Cap-50 hoặc cap-200, nếu cần, phải là train-only
ablation view sau khi có nhãn và sau group split; không được thay thế dedup
hoặc áp riêng theo sentiment.

## Kết quả phân hoạch của release 2.1.2

Release `lazada_vi_absa_curation_v2_1_2_20260725`, ID
`lazada-vi-absa-curation-e42d6c5319faedd3`, rule 2.1.2/schema 2.1, bao phủ
một-một toàn bộ parent:

| Frozen identifier | Giá trị |
|---|---|
| `built_at` | `2026-07-25T12:37:36.389780+00:00` |
| `source_cutoff_at` | `2026-07-25T07:15:13.282133+00:00` |
| Release manifest SHA-256 | `4a45a3b83f1bebf9d87751a61aaab168b3d957321935403ce06eb4477536128e` |
| Parent manifest SHA-256 | `3fed426fab18fe3887bfcac2e6e4996685b15561276549d7de05e51a10f28a43` |
| Source inventory SHA-256 | `87781ebbc60ba84c638991864f03a4907983cb93a80fa5ed5f22a02bd5e343c2` |
| Code inventory SHA-256 | `1f010e33fed37509ed976bb81c9467fb7036416abead096c5937d0d3f29fb922` |
| Guideline SHA-256 | `58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9` |

| Trạng thái | Số dòng | Cách hiểu |
|---|---:|---|
| `KEEP` | 20.429 | Text gốc đi vào clean-core kỹ thuật |
| `KEEP_CLEANED` | 193 | Derived text đã biến đổi có ledger và còn đạt quality |
| `QUARANTINE` | 11.166 | Cần con người quyết định keep/exclude/escalate |
| `EXCLUDE_AUTO` | 140 | Chỉ duplicate đã xác nhận; không xóa khỏi ledger |
| **Tổng** | **31.928** | Bằng đúng parent, không overlap và không thất thoát |

Clean-core kỹ thuật có 20.622 dòng. Hàng đợi curation có 11.306 dòng, gồm cả
11.166 quarantine và 140 duplicate để con người vẫn có thể audit quyết định
tự động. Annotation index có 20.622 dòng; pilot candidate có 450 dòng; clean
audit và template-calibration có 1.000 dòng mỗi file. Tất cả ô nhãn vẫn blank:
main annotation chưa được sinh.

Primary reason là `KEEP` 20.429, `PLATFORM_TEMPLATE` 9.874,
`REWARD_DISCLOSURE` 661, `INTERNAL_REPETITION` 348,
`POST_CLEAN_QUALITY` 207, `NONREVIEW_SUSPECT` 151, `DUPLICATE` 141,
`HARD_NONREVIEW` 59, `NEAR_DUPLICATE` 34 và `PRIVACY` 24. Primary reason phủ
một-một, còn reason-code là multi-label và không được cộng cơ học. Ví dụ release
có 9.575 structural catalogue, 1.848 global-template candidate, 1.319
product-template high, 588 post-clean-quality fail, 300 reward suspect, 139
keyboard smash, 41 advertisement, 7 Tagalog-dominant và 2 no-evaluation.

Phân bố clean-core vẫn lệch mạnh: 19.205/20.622 dòng là 5 sao; 431 dòng 1 sao,
146 dòng 2 sao, 256 dòng 3 sao và 584 dòng 4 sao. Rating không được dùng để
suy nhãn ABSA. Transport gồm 18.599 `requests_cookie`, 2.014 `selenium_dom` và
9 `requests`. Clean-core trải trên 971 product ID và 593 seller ID không rỗng;
225 dòng không có seller ID. Product lớn nhất đóng góp 297 dòng (1,44%) và
top-10 product đóng góp 1.981 dòng (9,61%). Vì vậy split sau annotation phải
group theo product/duplicate/template family, không được random theo row.

## Validator, checksum và khả năng tái lập

Validator fail closed đã chạy lại ngày 25/07/2026 và trả `VALID`. Audit kỹ
thuật độc lập xác nhận:

- checksum closure của 25 entry (24 artifact cộng `manifest.json`); thư mục
  có 26 file khi tính cả `SHA256SUMS.txt`; 49/49 file parent và 1.573/1.573
  source-inventory entry còn tồn tại, khớp SHA-256;
- 6/6 file code provenance, config, guideline, parent manifest và source
  inventory khớp byte với bản đã đóng gói;
- parent coverage đúng 31.928 dòng và bốn partition không giao nhau;
- raw hash/curated hash đúng cho từng record; transformation ID liên tục,
  input/output hash chain kín và không lưu raw PII span trong transformation
  ledger;
- 140/140 duplicate alias nằm ngoài clean-core, gồm 83 cross-transport và 57
  punctuation-exact; trong 57 punctuation alias có 36 cùng product và 21 khác
  product; representative của cả 140 alias đều được giữ;
- clean-core có 0 exact-text duplicate, 0 punctuation-normalized duplicate và
  không trùng cặp `(product_id, review_id)`;
- 80/80 QC exact ID nằm đúng một lần trong review queue, đều
  `QUARANTINE`/`MANUAL_PENDING` và không annotation-eligible;
- post-clean quality, structural-template evidence và internal-repetition
  evidence có thể tính lại; annotation index ánh xạ một-một với 20.622 dòng
  clean-core.

Rebuild-check dựng release vào thư mục tạm với cùng `built_at`, so sánh 25
checksum entry và trả `BYTE_REPRODUCIBLE`. Bộ test được discover trong active
suite hiện tại đạt 74/74 và `py_compile` đạt. `git diff --check` cũng đã chạy
nhưng không bao phủ file
untracked, nên không được xem là validation đầy đủ cho toàn worktree. Config
SHA-256 hiện hành là
`c3de03b767183c6e19bcfd58a60885f9582555478bbd73cc8bd851b1955dee57`.

Code inventory của release chỉ khóa sáu file code và chưa
bao gồm `tests/test_release_builder.py` cùng
`tests/test_collector_core.py`, dù chúng tham gia bộ 74 test. Worktree hiện
chưa có commit/tag khóa snapshot mới. Do đó `BYTE_REPRODUCIBLE` chứng minh khả
năng tái dựng từ sáu file đã inventory trong workspace hiện tại, chưa đủ để
khẳng định reproducibility ở cấp repository lâu dài. Release tiếp theo phải
mở rộng code inventory, lưu test runner/environment lock và gắn commit/tag.
Config và guideline là hai provenance artifact được copy/hash riêng, không
nằm trong sáu entry của `CODE_SHA256SUMS.txt`.

## Kiểm toán ngữ nghĩa release 2.1.2

Một holdout 250 dòng được chọn deterministic, tỷ lệ theo
category×rating×transport từ clean-core 2.1.2. Tập chọn loại các ID có thể
khôi phục từ audit, calibration, QC và holdout trước; 2.377 ID nằm trong union
loại, trong đó 1.159 thuộc clean-core hiện hành, để lại 19.463 candidate.
Salt là `v212-unseen-semantic-holdout-v1\0`; SHA-256 của tập sample là
`486b9591ca0fa37cc7720a14d18b4ca2a9d3bae49b5790662ee291e35b935e5c`,
tính trên các `sample_id` sort tăng, nối bằng LF và không có terminal LF.
Full 250-row decision ledger được persist tại
`docs/audits/v2_1_2_semantic_holdout_250.csv`, chỉ lưu text hash, nhãn
operational và note; file SHA-256 là
`d54196204393a740b5f0d2acc023d02b6f759c6ab572383a9e49dfb038a72b57`.
Danh sách ID của một số structural calibration panel cũ không được lưu đầy đủ,
vì vậy không thể chứng minh overlap bằng 0 với các panel đó; đây là hạn chế
provenance phải báo cáo.

Kết quả content-only do một analyst/agent thực hiện:

| Nhãn audit | Số dòng | Tỷ lệ trên 250 |
|---|---:|---:|
| Strict ABSA-usable | 230 | 92,0% |
| Pure catalogue | 11 | 4,4% |
| Hard non-review | 3 | 1,2% |
| Foreign/gibberish | 2 | 0,8% |
| Borderline | 4 | 1,6% |

Wilson 95% của strict-usable trong chính sample là 87,97--94,76%. Trong 230
dòng usable còn 23 dòng mixed-noisy (10,0% của usable; 9,2% toàn sample).
Sau unblind, sample có rating 5/4/3/2/1 lần lượt 243/4/1/0/2 và transport
cookie/DOM là 226/24; toàn bộ 20 dòng không strict đều là 5 sao. Kết quả chỉ
ước lượng eligible pool 19.463 dòng sau exclusion, không phải toàn core. Wilson
interval coi observation độc lập và không phản ánh dependence theo
product/template, tuning history hoặc sai số một người đọc. Kết quả nhất quán
với screening tốt hơn trong chính panel hiện tại, nhưng các panel một analyst
không hoàn toàn so sánh được. Đây **không** phải human gold, không đo độ đúng
nhãn ABSA và không đủ để gọi corpus final.
Nếu 20 lỗi được dùng để tạo rule 2.1.3 thì holdout này trở thành calibration;
release kế tiếp phải có holdout mới.

Audit delta 2.1.1→2.1.2 xác nhận 1.769/22.391 dòng core cũ chuyển sang
quarantine, không có dòng mất hoặc auto-exclude. SHA-256 của sorted delta ID
là `939c8ad950674a435c2aad56cf13d101caa365a8430dc4c579bac4493cea91f6`.
Trong sample 200 dòng chọn bằng salt `v212-delta-audit-v1\0`, analyst/agent
đánh giá 156 pure catalogue, 37 mixed catalogue+review, 5 borderline và 2
natural-clean: 193/200 (96,5%; Wilson 92,95--98,29%) được xếp operationally
quarantine-appropriate bởi analyst này, còn false-positive natural-clean là
2/200 (1,0%; Wilson
0,27--3,57%). Hai ID natural-clean là
`lzv1-5084a2f367b98534defff308` và
`lzv1-e8ebed2fc0ca60a6e04f34b9`. Không nới threshold toàn cục vì một blanket
buyer/UI veto sẽ miễn tới 459/1.638 hit MODULAR và làm giảm mạnh recall
catalogue; hai case này phải đi human override hoặc một rule mới được kiểm trên
holdout khác.

Full 200-row sampled-audit ledger đã được lưu trong workspace, không kèm review
nguyên văn, tại
`docs/audits/v2_1_2_delta_audit_200.csv`; file SHA-256 là
`a1c8a494c48b67cd4fb0e2333b79009998791cf3c5478d1db2d3fe521eeaa943`.
Ordered sample, sorted sample-set và rank-order label hashes lần lượt là
`250e154bdfa9ae051bb87294281f50721ba0df9610f7d17e649a83a599a04c9a`,
`515f2183f7a81b58a9558d336f5430ba9971c9db5370836a168c05e01a9adbb1`
và `61ec26e7d1a9f29d4d2dc83684dd64e1633609310dd341d7ae9408e88480db93`.

## Lịch sử snapshot và giới hạn khẳng định

- Rule 2.0.0 có 26.078 clean-core; semantic audit 200 dòng chỉ tìm thấy
  180/200 ABSA-usable và ước lượng post-stratified chẩn đoán khoảng 82%.
- Rule 2.1.0 có 23.935 clean-core; audit cân bằng 200 dòng tìm thấy 192/200
  usable, 5 pure catalogue, 3 borderline và 20/192 mixed-noisy. Wilson
  92,31--97,96% chỉ áp dụng cho sample đã dùng để hiệu chỉnh.
- Rule 2.1.1 có 22.391 clean-core; unseen audit 200 dòng tìm thấy 179/200
  strict usable và được dùng để xây 2.1.2, nên snapshot bị supersede.
- Rule 2.1.2 có 20.622 clean-core; đã đạt validator/rebuild-check và có holdout
  agent-QC mới, nhưng vẫn là **pre-annotation curation release**.

File `curation_audit_1000.csv`, `template_calibration.csv`,
`curation_review_queue.csv` và `pilot_candidate.csv` đang chứa quyết định/nhãn
trống. Chưa được gọi corpus này là gold/final cho tới khi hoàn tất curation
người thật, pilot double-blind, IAA trước adjudication, adjudication và final
quality gate.

## Quyền riêng tư, pháp lý và điều kiện công bố

Mọi `source_url` quan sát đều thuộc `www.lazada.vn`; collector dừng khi gặp
CAPTCHA/challenge và không cài cơ chế bypass. Có 197 run đóng ở trạng thái
`paused_rate_limit`. Giá trị cookie không được ghi vào review hoặc release,
nhưng manifest có `cookie_session` metadata/count và các run DOM ghi
`browser_profile_may_persist=true`. Local `src/cookies.txt` hiện là một browser
cookie export rộng, có credential cho cả domain ngoài phạm vi Lazada. File đã
được `.gitignore` bắt bởi pattern `**/*cookie*.txt`, nhưng vẫn là secret trên
máy: không được commit, đóng gói, chia sẻ hoặc trích nội dung; cần rotate và
xóa an toàn sau thu thập theo quyết định của chủ dữ liệu. Audit schema release
không thấy các khóa buyer/avatar/cookie, email, phone, token hay username; 39
span phone/email/URL đã được thay bằng placeholder trong derived text và
ledger chỉ lưu hash/offset.

Tuy vậy, local partition vẫn giữ nguyên văn review, review/product/seller ID và
source URL để truy xuất khoa học. Repository không chứa giấy phép tái phân
phối, snapshot Terms of Service, legal opinion, IRB/ethics approval hay consent
của người viết review. Do đó không được tuyên bố dữ liệu là public domain,
được Lazada cấp phép, chắc chắn tuân thủ ToS hoặc đã được ethics board phê
duyệt. Trước khi công bố ra ngoài nhóm nghiên cứu phải có legal/terms/privacy
review riêng và tạo artifact tối thiểu hóa định danh; file annotation blind
không chứa rating, product/seller ID, transport, URL hoặc model prediction.
Tuy nhiên annotation CSV vẫn giữ nguyên văn review có thể tìm kiếm, nên
“blinded metadata” không loại hết re-identification risk.

# Quy trình annotation ABSA

## Đơn vị và bằng chứng

Annotator đọc toàn review và gán một nhãn cho từng aspect. Nhãn phải dựa trên
bằng chứng hiển thị trong text, không suy đoán theo rating. Shipping, packaging
và service vẫn có giá trị ngay cả khi người dùng nói chưa sử dụng sản phẩm.

Multi-polarity `1, -1` chỉ dùng khi cùng một aspect có cả bằng chứng tích cực
và tiêu cực, ví dụ đóng gói cẩn thận nhưng hộp bị móp. Hai polarity thuộc hai
aspect khác nhau không tạo nhãn mixed cho từng aspect.

## Pilot và hiệu chỉnh guideline

Trước annotation toàn bộ, chọn pilot đa dạng theo category, rating, độ dài và
quality flags. Ít nhất hai annotator gán độc lập; một phần pilot được ba người
gán. Các disagreement phải được phân nhóm thành lỗi guideline, lỗi boundary,
nhầm aspect hoặc polarity conflict rồi cập nhật guideline trước vòng chính.

Pilot được tổ chức thành hai vòng, mỗi vòng 100--200 review. Annotator chỉ qua
qualification khi mention macro-F1 với expert gold đạt ít nhất 0,85, 5-state
macro-F1 đạt ít nhất 0,80 và exact agreement với gold đạt ít nhất 0,85.
Ngưỡng này phải được đăng ký trước khi xem kết quả annotation chính.

## Main annotation và quality control

Một trăm phần trăm clean-core được double-annotate độc lập và blind. Annotator
không được xem rating, metadata sản phẩm, dự đoán model hoặc nhãn của người
còn lại nếu những trường đó không phải input của model khi inference. Batch có
300--500 review để giảm fatigue.

Khoảng 5% hidden expert-gold và 2% repeated control được chèn vào luồng làm
việc nhưng không tạo sample mới trong corpus. Mỗi annotation event lưu
append-only: `sample_id`, exact text hash, pseudonymous annotator ID,
guideline version, timestamp, chín raw labels, evidence clause và uncertainty
code. Không overwrite lần gán nhãn trước.

Batch chỉ được nhận khi:

- đủ 9/9 nhãn và không còn blank;
- text hash khớp release;
- hidden-gold exact agreement ít nhất 0,85;
- hidden-gold macro-F1 ít nhất 0,80;
- repeated-control exact agreement ít nhất 0,90.

Nếu batch không đạt, dừng assignment của annotator, audit nguyên nhân và gán
lại từ batch đạt gate gần nhất. Thời gian annotation ngắn chỉ là tín hiệu mở
audit, không phải lý do tự động loại annotator.

## Inter-Annotator Agreement

Báo cáo agreement theo từng aspect và toàn bộ:

- Krippendorff's alpha hoặc Fleiss' kappa cho nhiều annotator;
- Cohen's kappa cho cặp annotator khi phù hợp;
- agreement về aspect presence;
- macro-F1 giữa annotator cho polarity;
- tỷ lệ exact-match trên vector chín aspect.

Không chỉ báo cáo phần trăm agreement vì lớp `2` có thể chiếm ưu thế.

IAA chính được tính trên hai annotation blind **trước adjudication**. Expert
adjudicator nhìn thấy bất đồng nên không được coi là annotator độc lập thứ ba
trong Fleiss' kappa. Nếu cần IAA ba người, một subset phải được người thứ ba
gán blind trước khi xem A/B.

Báo cáo tối thiểu:

- raw exact agreement và Gwet AC1 hoặc Krippendorff alpha nominal trên năm
  trạng thái cho từng aspect;
- mention macro-F1 và AC1 trên `{mentioned, absent}`;
- polarity agreement có điều kiện khi cả hai cùng xác định aspect mentioned;
- exact-set/Jaccard trên tập `{negative, positive, neutral}`;
- mixed-specific precision, recall, F1 và support;
- 95% confidence interval bootstrap theo product/template group.

Gate đề xuất: mention macro-F1 ≥ 0,85 và AC1 ≥ 0,80; full five-state exact
agreement ≥ 0,80; polarity alpha/AC1 overall ≥ 0,80; không aspect phổ biến nào
dưới 0,67. Mixed phải được adjudicate 100% và luôn báo support.

## Adjudication và gold labels

Disagreement được adjudicate bởi người thứ ba hoặc hội đồng. Lưu nhãn trước
adjudication, nhãn cuối, reason và guideline version. Gold test set phải được
double/triple-annotated và adjudicated; không dùng nhãn LLM chưa kiểm chứng làm
gold.

Expert adjudicate 100% disagreement, 100% uncertainty, neutral/mixed hiếm và
một random sample 5% các hàng hai annotator đồng thuận. Nếu một quyết định tạo
quy tắc guideline mới, tăng guideline version và truy hồi mọi review bị ảnh
hưởng thay vì chỉ áp dụng từ batch sau.

# Split, augmentation và mô hình

## Chống leakage

Chưa chốt train/dev/test trước khi có nhãn. Khi split:

1. Nối record bằng product ID, record-duplicate cluster và template family.
2. Toàn connected group phải nằm trong một split.
3. Tối ưu phân bố category và label ở cấp group.
4. Gold test được khóa trước model selection.
5. Báo cáo số product và review trong từng split.

## Augmentation

Chỉ augment train sau split. Synthetic sample phải có `source_type`,
generator, prompt version, parent ID và quality review. Không augment dev/test,
không dùng synthetic để che giấu thiếu dữ liệu rare class, và phải có ablation
real-only so với real-plus-synthetic.

## Metric ABSA

Bộ metric tối thiểu:

- macro/micro/weighted F1 theo từng aspect;
- macro-F1 ưu tiên vì class imbalance;
- precision/recall cho negative và mixed;
- exact match của vector chín aspect;
- Hamming loss nếu biểu diễn multi-label;
- confusion matrix theo aspect;
- bootstrap confidence interval trên test group;
- performance theo category, rating và độ dài;
- error analysis cho negation, implicit aspect và multi-polarity.

Primary end-to-end metric được định nghĩa trên tập tuple
`{(aspect, polarity)}`. Mixed sinh hai tuple positive và negative. Báo joint
aspect--polarity micro-F1 và macro-F1 trên 27 aspect×polarity label, sample
Jaccard và exact-set match. Accuracy đơn lẻ không được dùng làm metric chính.

Một implementation model phù hợp dùng mention head `M[N,9]` và polarity head
multi-hot `S[N,9,3]` theo thứ tự tensor `[negative, positive, neutral]`.
Sentiment loss chỉ tính ở gold-mentioned aspects. Neutral loại trừ positive và
negative; mixed là hai bit negative+positive, không phải ép thành một class làm
mất cấu trúc đa cực. Threshold chỉ được tune trên dev rồi khóa trước test.

## Cảnh báo đối với code legacy

Loader lịch sử có nhánh diễn giải blank/NaN thành absent `2` và có thể ép nhãn
lỗi thành neutral. Do đó tuyệt đối không train trực tiếp từ CSV annotation dở
dang. Final-label validator phải fail closed khi gặp blank hoặc biến thể nhãn
không canonical. Random row-level fold của code cũ gây product/template
leakage; `combined_score` có trọng số thủ công chỉ là metric phụ.

# Artifact hiện hành và còn thiếu

Các artifact sau **nằm trong frozen release 2.1.2** và được bao phủ bởi 25
checksum entry:

- `curation_records.jsonl`, `clean_core.jsonl`, `quarantine.jsonl`,
  `excluded.jsonl`;
- `duplicate_aliases.jsonl`, `cross_transport_candidates.jsonl`,
  `near_duplicate_candidates.jsonl`, `template_families.jsonl`;
- `transformations.jsonl` và per-record curation decision/evidence;
- `annotation/index.csv`, `pilot_candidate.csv`,
  `curation_audit_1000.csv`, `template_calibration.csv`,
  `curation_review_queue.csv` và schema blank;
- versioned annotation guideline, data card, parent/source/code provenance,
  `manifest.json` và checksum list. Bản thân `SHA256SUMS.txt` là danh sách
  kiểm tra nên không tự liệt kê chính nó.

Các bằng chứng **nằm ngoài frozen release** nhưng hiện đã được lưu trong
workspace:

- `docs/audits/V2_1_2_AUDIT_REPORT.md`;
- `docs/audits/v2_1_2_semantic_holdout_250.csv`;
- `docs/audits/v2_1_2_delta_audit_200.csv`;
- `docs/audits/LEGACY_DATASET_AUDIT.md` và
  `docs/audits/legacy_xlsx_sha256.csv`;
- living protocol Markdown và DOCX.

Các file ngoài release có hash riêng được ghi trong protocol/audit nhưng chưa
nằm trong commit/tag. Năm audit artifact được khóa chung bởi
`docs/audits/SHA256SUMS.txt`; protocol/DOCX không nằm trong closure này. DOCX
chỉ được coi hiện hành sau khi TASK-014 render lại và đọc ngược thành công.

Các artifact sau **chưa tồn tại hoặc chưa hoàn tất** và là gate bắt buộc của
final/gold release:

- human curation decision cho queue/audit và reviewer/adjudicator provenance;
- annotation CSV đã điền đủ 9/9 aspect, không còn blank;
- hai annotation blind trước adjudication, uncertainty/evidence ledger và
  final adjudication ledger;
- IAA report với confidence interval, qualification/pilot report;
- group-constrained train/dev/test manifest và leakage audit;
- final-label validator/checksum closure sau annotation;
- model-ready tensor export có kiểm schema và benchmark/ablation report;
- quyết định legal/terms/privacy cho phạm vi phân phối dự kiến.

# Nhật ký tác vụ

## TASK-20260725-001 — Khởi tạo living protocol và cơ chế ghi bắt buộc

- **Trạng thái:** Đã thực hiện và xác thực.
- **Mục tiêu:** Tạo một tài liệu có thể dùng làm nguồn viết Methodology,
  Data Collection, Data Cleaning và Experimental Protocol của paper; bảo đảm
  mọi tác vụ tiếp theo được ghi lại.
- **Đầu vào:** Trạng thái repository ngày 2026-07-25, release v1 và kết quả
  audit duplicate/non-review.
- **Thao tác:** Tạo bản nguồn Markdown, script render DOCX bằng Pandoc và
  `AGENTS.md` quy định cập nhật tài liệu sau mọi tác vụ.
- **Đầu ra dự kiến:** `docs/DATASET_CONSTRUCTION_PROTOCOL.docx`.
- **Kiểm tra:** DOCX phải là OOXML hợp lệ, chứa `word/document.xml`, và Pandoc
  phải đọc ngược được tối thiểu 1.000 ký tự plain text.
- **Quyết định:** Raw luôn bất biến; release v1 không được dùng để annotation;
  mọi clean release mới phải có ledger và checksum.
- **Hạn chế:** Tài liệu mới khởi tạo; các counts của clean release v2 chưa tồn
  tại và phải được bổ sung sau khi pipeline thực thi.
- **Phụ thuộc kế tiếp:** Chốt taxonomy và triển khai clean release v2.

## RETRO-20260725-A — Thu thập, đóng crawl và release kỹ thuật v1

- **Trạng thái:** Tóm tắt hồi cứu từ manifest và artifact đã xác thực.
- **Kết quả:** 31.949 raw review rows; 31.928 canonical V2; 393 manifest;
  1.573 source-inventory entry (1.572 crawl artifact cộng `.gitkeep`); không
  còn run đang chạy.
- **Ý nghĩa:** Chứng minh count/provenance kỹ thuật, không chứng minh semantic
  validity.

## RETRO-20260725-B — Audit semantic duplicate và non-review

- **Trạng thái:** Đã thực hiện read-only.
- **Kết quả chính:** phát hiện hybrid identity gap, comma-template pollution,
  system/promo/off-topic text và reward filler; dừng annotation release v1.
- **Quyết định:** Không xóa raw, không tiếp tục gán nhãn batch v1, xây release
  sạch mới với các tầng keep/quarantine/exclude.

## TASK-20260725-002 — Đặc tả annotation, IAA, split và đánh giá ABSA

- **Trạng thái:** Đã đặc tả; chưa thực hiện annotation.
- **Mục tiêu:** Chuyển schema legacy thành một protocol ABSA đa cực có quality
  gate, chống leakage và metric phù hợp để bảo vệ trong paper.
- **Đầu vào:** Chín aspect, năm trạng thái canonical, guideline lịch sử, audit
  code loader/evaluation cũ.
- **Thao tác:** Xác định đơn vị ACSA cấp review; formalize label encoding;
  thiết kế 100% double-blind annotation, hidden-gold/repeated controls, IAA
  trước adjudication, expert adjudication và group-constrained split.
- **Kết quả:** Protocol quy định rõ blank khác absent, tensor polarity theo
  `[negative, positive, neutral]`, mixed là hai polarity bits, IAA không dùng
  nhãn sau hòa giải, và test không chứa synthetic data.
- **Quyết định:** Chưa tạo train/dev/test khi chưa có nhãn final. Augmented
  data chỉ được dùng train-only trong ablation. Primary metric là joint
  aspect--polarity F1 thay vì accuracy hoặc combined score.
- **Hạn chế:** Ngưỡng qualification/IAA là preregistered proposal; cần pilot
  thật để xác nhận độ khả thi và báo cả giá trị đo được, không chỉ threshold.
- **Phụ thuộc kế tiếp:** Clean release v2 và blank pilot package đã qua
  schema/curation gate; yêu cầu “không blank” chỉ áp dụng sau final annotation.

## TASK-20260725-003 — Chốt taxonomy và cấu hình cleaning v2

- **Trạng thái:** Đã thực hiện ở thời điểm đặc tả; pipeline khi đó chưa chạy.
  Policy này là mốc v2.0 lịch sử và đã được siết ở 2.1+.
- **Mục tiêu:** Chuyển audit định tính thành quy tắc deterministic, versioned,
  không xóa nhầm review ABSA hợp lệ.
- **Đầu vào:** 31.928 canonical review, các audit duplicate/API--DOM,
  non-review, comma-template, reward disclosure và code quality hiện hành.
- **Thao tác:** Tạo `configs/cleaning_v2.json`; chốt bốn trạng thái cuối
  `KEEP`, `KEEP_CLEANED`, `QUARANTINE`, `EXCLUDE_AUTO`; xác định precedence,
  threshold, bilingual heading aliases, hard signatures, PII patterns,
  product cap và annotation parameters.
- **Quy tắc an toàn lịch sử:** v2.0 từng cho phép `EXCLUDE_AUTO` với duplicate
  chắc chắn hoặc artifact/non-review gần như chắc chắn. Từ 2.1+, policy đã
  siết thành duplicate xác nhận duy nhất; mọi heuristic nội dung chỉ
  quarantine.
  Product cap 50 cũng bị tắt ở canonical release và chỉ còn là train-only
  ablation dự kiến. Frequency/similarity trung gian không auto-delete.
- **Threshold lịch sử của config 2.0.0 tại thời điểm task:** cross-transport
  J5 ≥ 0,50 trong block metadata chặt;
  template global có ít nhất hai recurrent clause, review-DF ≥ 5,
  product-DF ≥ 3 và coverage ≥ 0,55; product-local review-DF ≥ 10 và coverage
  ≥ 0,70; product cap 50 bằng stable hash seed, không dựa vào rating. Đây là
  mốc lịch sử, không phải threshold template hiện hành của 2.1.2.
- **Đầu ra:** Config JSON là nguồn duy nhất cho rule version 2.0.0. Mọi output
  phải ghi hash config.
- **Validation cần thực hiện:** JSON parse; invariant bốn status phủ đúng
  31.928 record; manual sample theo mỗi reason; byte-stable rerun; raw hash
  không đổi.
- **Hạn chế:** Candidate template theo frequency chưa phải bằng chứng đủ để
  auto-delete; mọi candidate phải ở quarantine hoặc có lexicon được duyệt.
- **Phụ thuộc kế tiếp:** Cài đặt detector, transformation ledger và builder.

## TASK-20260725-004 — Cài đặt và kiểm thử thư viện curation thuần hàm

- **Trạng thái:** Đã thực hiện và xác thực.
- **Mục tiêu:** Cài đặt các phép chuẩn hóa, đối sánh và biến đổi văn bản dưới
  dạng deterministic pure functions trước khi cho phép pipeline chạm vào
  release dữ liệu.
- **Phạm vi:** `src/lazada_collector/curation.py`,
  `tests/test_curation.py`, `configs/cleaning_v2.json`.
- **Thao tác đã thực hiện:** Cài đặt Unicode NFKC/word key/punctuation key;
  chuẩn hóa ngày và SKU song ngữ API--DOM; canonical heading; word 5-gram
  Jaccard; complete AllPairs candidate join với exact verification; connected
  components và chọn representative; tách clause có cả dấu phẩy; collapse
  clause lặp; tách reward disclaimer thuần; phát hiện hard signature; redaction
  email/điện thoại/URL; language/gibberish flags; clause document frequency;
  template evidence; deterministic stratified sampling. Bổ sung đầy đủ cấu
  hình quality sau biến đổi vào JSON thay vì dựa vào default ngầm.
- **Các lỗi được test phát hiện và sửa:** underscore làm mất token;
  chuỗi SKU DOM ghép nhiều field không có dấu phẩy; literal `\n` không được
  phát hiện; `zip()` có thể làm mất record khi hai input frequency lệch độ dài;
  và nhiều clause lặp liền nhau có thể bị gộp span evidence.
- **Validation:** `python -X utf8 -m unittest tests/test_curation.py -v` chạy
  24/24 test curation thành công; `python -X utf8 -m unittest discover -s
  tests -p "test_*.py" -v` chạy 52/52 test discoverable trong active suite
  thời điểm đó; `py_compile` thành công. `git diff --check` không phát hiện
  lỗi ở tracked diff nhưng không bao phủ file untracked, nên không phải bằng
  chứng whitespace cho toàn module/config/test/guideline. Venv không cài
  `pytest`, vì vậy
  `unittest` hiện hữu của repository được dùng làm runner, không cài thêm
  dependency chỉ để đổi test runner.
- **Quyết định:** Similarity/frequency chỉ tạo evidence; không tự xóa
  near-duplicate hay template mơ hồ. Transformation không ghi nội dung span bị
  cắt mà ghi hash, offset, số ký tự, input hash và output hash để giảm rủi ro
  lộ PII đồng thời vẫn replay/audit được.
- **Đầu ra:** Một tầng thuật toán độc lập với I/O, đủ để unit-test và tái sử
  dụng trong release builder/validator.
- **Hạn chế:** Unit test chứng minh hành vi thuật toán trên ca kiểm soát, chưa
  thay thế corpus-level calibration và human false-positive audit.
- **Phụ thuộc kế tiếp:** Chạy builder trên đủ 31.928 candidate và hiệu chỉnh
  bằng thống kê/audit mẫu.

## TASK-20260725-005 — Viết guideline gán nhãn ABSA phiên bản 2

- **Trạng thái:** Đã thực hiện ở mức tài liệu; annotation chưa bắt đầu.
- **Mục tiêu:** Tách guideline vận hành cho annotator khỏi phần mô tả phương
  pháp tổng quát, khóa rõ ontology và loại bỏ cách diễn giải nhãn tùy ý.
- **Đầu ra:** `docs/ABSA_ANNOTATION_GUIDELINE_V2.md`, 866 dòng tại thời điểm
  tạo, mô tả đủ chín aspect, năm trạng thái legacy, ví dụ ca khó,
  multi-polarity, evidence clause, uncertainty, double-blind pilot, IAA trước
  adjudication và final QC fail-closed.
- **Schema đã khóa:** blank chỉ có nghĩa “chưa gán”; `2` là aspect không xuất
  hiện; `1, -1` chỉ áp dụng khi cùng một aspect có cả positive và negative.
  Tensor model-ready giữ đúng thứ tự code hiện tại
  `[negative, positive, neutral]`.
- **Blindness:** File annotator nhìn thấy không chứa rating, product/seller ID,
  transport, model prediction hoặc nhãn của annotator còn lại.
- **Validation:** Markdown UTF-8 tồn tại, 866 dòng. `git diff --check` không
  báo lỗi ở tracked diff nhưng không kiểm file guideline untracked tại thời
  điểm đó.
- **Hạn chế:** Guideline cần được hiệu chỉnh dựa trên disagreement của pilot,
  nhưng mọi thay đổi phải tăng version và truy hồi các review bị ảnh hưởng.
- **Phụ thuộc kế tiếp:** Chỉ phát hành pilot từ clean-core đã qua validator;
  chưa phát hành main annotation như thể pilot đã đạt gate.

## TASK-20260725-006 — Dựng và kiểm định snapshot curation rule 2.0.0

- **Trạng thái:** Đã thực hiện và xác thực kỹ thuật; snapshot đã bị thay thế.
- **Mục tiêu:** Chạy curation end-to-end lần đầu trên toàn bộ parent thay vì
  chỉ unit-test từng rule; tạo partition, ledger, annotation package và
  checksum closure.
- **Đầu vào:** 31.928 canonical record của
  `lazada_vi_reviews_v1_20260725`, config cleaning rule 2.0.0, guideline V2 và
  source inventory bất biến.
- **Thao tác:** Cài đặt release builder và fail-closed validator; tạo
  cross-transport aliases, punctuation-text aliases, transformation ledger,
  template evidence, clean-core, quarantine, exclusion, audit/calibration
  samples và provenance inventories.
- **Kết quả đo được:** 25.255 `KEEP`, 823 `KEEP_CLEANED`, 5.710
  `QUARANTINE`, 140 `EXCLUDE_AUTO`; clean-core 26.078. Bốn partition phủ đúng
  31.928 parent record. Chỉ 140 duplicate đã xác nhận được auto-exclude.
- **Quyết định:** Không phát hành snapshot này cho annotator trước semantic
  audit; validator cấu trúc không đủ chứng minh content validity.
- **Hạn chế:** Template detector ban đầu dựa nhiều vào recurrent-clause
  coverage và bỏ lọt danh sách catalogue được paraphrase hoặc ghép clause.
- **Phụ thuộc kế tiếp:** Đọc content-blind sample và audit false negative
  template/non-review.

## TASK-20260725-007 — Semantic audit snapshot 2.0.0 và quyết định supersede

- **Trạng thái:** Đã thực hiện; kết quả chỉ là audit chẩn đoán, không phải
  human gold annotation.
- **Mục tiêu:** Đo xem clean-core kỹ thuật có thực sự chứa review ABSA hữu ích
  hay còn catalogue, quảng cáo và văn bản không phải review.
- **Đầu vào:** Mẫu 200 dòng từ clean-core 26.078, nội dung review được đọc mà
  không dùng rating để suy polarity.
- **Phương pháp:** Phân loại operational thành ABSA-usable, pure
  catalogue/template, mixed/noisy, hard non-review và borderline; xem riêng
  các strata rating để phát hiện sai lệch mẫu. Các case lỗi được truy ngược về
  clause recurrence và structural evidence.
- **Kết quả:** 180/200 dòng được đánh giá ABSA-usable. Ước lượng
  post-stratified theo rating khoảng 82%, nhưng không phải confidence interval
  đại diện cho corpus vì mẫu và rule đã được dùng để chẩn đoán. Nhiều dòng
  danh sách tính năng vẫn vượt substantive quality.
- **Quyết định:** Đánh dấu release
  `lazada_vi_absa_curation_v2_20260725` là superseded; không dùng cho
  annotation hoặc tuyên bố final dataset.
- **Hạn chế:** Một người/agent đọc chẩn đoán không thay thế double-blind human
  audit; tỷ lệ 82% không được dùng làm số chất lượng cuối.
- **Phụ thuộc kế tiếp:** Structural catalogue detector, negative controls và
  holdout rule calibration.

## TASK-20260725-008 — Hiệu chỉnh structural/template rules và snapshot 2.1.0

- **Trạng thái:** Đã thực hiện; snapshot 2.1.0 sau đó bị thay thế bởi 2.1.1.
- **Mục tiêu:** Bắt catalogue list bằng cấu trúc thay vì keyword đơn, đồng
  thời bảo vệ review có trải nghiệm người mua, complaint và structured aspect
  value.
- **Thao tác:** Bổ sung segment/delimiter/short-ratio/marketing-opener,
  recurrent weak clause, trailing delimiter và buyer-residual evidence; giữ
  dấu tiếng Việt cho buyer anchors; bỏ `nhỏ|nho` khỏi weak informal-buyer
  tokens; thêm positive/negative regression tests.
- **Calibration đã chạy:** Tập 200 bắt 17/17 pure catalogue đã biết và không
  bắt 178/178 review usable. Holdout rule 120 dòng cho 9/9 precision trên
  candidate, 9/13 recall pure template, 0/98 false positive trên review sạch
  và 0/8 trên mixed buyer-plus-template. Các mẫu này dùng để hiệu chỉnh rule,
  không phải final quality estimate.
- **Lỗi tích hợp được phát hiện:** Một build trung gian bị validator chặn do
  builder tính internal-repetition evidence trước PII cleaning còn validator
  tính sau cleaning. Artifact trung gian không được phát hành; thứ tự tính đã
  được thống nhất và test hóa.
- **Kết quả snapshot 2.1.0:** ID
  `lazada-vi-absa-curation-ec0130191d702235`; 23.211 `KEEP`, 724
  `KEEP_CLEANED`, 7.853 `QUARANTINE`, 140 `EXCLUDE_AUTO`; clean-core 23.935.
- **Phụ thuộc kế tiếp:** Semantic audit mới và rebuild-check độc lập.

## TASK-20260725-009 — Kiểm định snapshot 2.1.0 và sửa lỗi tái lập README

- **Trạng thái:** Đã thực hiện; snapshot bị supersede sau QC.
- **Mục tiêu:** Kiểm tra cả tính toàn vẹn kỹ thuật, độ sạch ngữ nghĩa và khả
  năng dựng lại byte-for-byte.
- **Kết quả validator:** Partition, checksum, 140 aliases, transformation
  chain, annotation index và quality/template invariants đều đạt.
- **Semantic audit:** Mẫu cân bằng category×rating×transport 200 dòng có
  192/200 ABSA-usable, 5 pure catalogue và 3 borderline; 20/192 usable nhưng
  trộn nhiễu. Wilson 95% cho chính mẫu này là 92,31--97,96%. Vì mẫu được dùng
  để sửa rule và không tỷ lệ với corpus, con số không được trình bày như final
  unbiased corpus estimate.
- **Negative-control audit:** Con số 953 là union ở snapshot calibration trung
  gian; khi chạy toàn bộ rule 2.1.1, hai nhánh tương ứng có union 1.088 dòng.
  Trên mẫu kiểm định 500 dòng, 23/23 candidate đọc được đều là
  catalogue/template, chưa thấy natural/mixed false positive. Panel này đã
  tham gia hiệu chỉnh nên không phải final holdout.
- **Rebuild-check:** Lần dựng lại chỉ lệch `README.md` và kéo theo
  `manifest.json`, vì README ghi tên thư mục build tạm ngẫu nhiên. Builder đã
  được sửa để ghi release name ổn định; regression test xác nhận hai temp root
  khác nhau tạo README giống nhau.
- **Quyết định:** Không gọi snapshot 2.1.0 là final; chuyển mọi global template
  candidate sang quarantine và bổ sung các case lỗi semantic đã biết.
- **Phụ thuộc kế tiếp:** Rule 2.1.1, final rebuild-check và holdout chưa từng
  dùng để hiệu chỉnh.

## TASK-20260725-010 — QC rule 2.1.1 và dựng snapshot tiền gán nhãn

- **Trạng thái:** Đã thực hiện và xác thực kỹ thuật; snapshot sau đó bị
  supersede bởi 2.1.2.
- **Mục tiêu:** Loại khỏi clean-core các false negative còn thấy ở 2.1.0 mà
  không xóa vật lý hay dùng rating/sentiment để quyết định.
- **Code/config thay đổi:** Buộc `TEMPLATE_GLOBAL_CANDIDATE` vào quarantine;
  thêm `RECURRENT_SPEC_LIST`, `LOW_DENSITY_BROCHURE_LIST`, reward-voucher,
  quảng cáo pass-lại/size-sale/CTA, Tagalog-dominant, generic purchase advice
  và conservative keyboard-smash evidence. Mọi heuristic nội dung vẫn chỉ
  quarantine.
- **QC exact-ID:** 39 dòng audit được escalation có version trong config:
  11 pure template, 3 borderline và 25 mixed/noisy/missed-smash. Sáu catalogue
  paraphrase nằm trong nhóm pure-template. Đây là curation escalation, chưa
  phải human label và không được dùng để tuyên bố semantic paraphrase recall
  tổng quát.
- **Các lần dựng không phát hành:** Intermediate ID
  `lazada-vi-absa-curation-1012c06082d490ab` có 22.399 clean-core và 9.389
  quarantine; 26 file sinh tự động đã được xóa sau khi phát hiện thêm năm ID
  thuộc các cặp brochure paraphrase. Raw, parent và hai snapshot trước không
  bị sửa; release có thể tái dựng từ parent/config. Một build kế tiếp được
  dừng trước publish để thêm Tagalog/advice rules và không để lại output.
- **Final build đã chạy:** Release
  `lazada_vi_absa_curation_v2_1_1_20260725`, ID
  `lazada-vi-absa-curation-a1efc1c16f0e8f02`; 21.858 `KEEP`, 533
  `KEEP_CLEANED`, 9.397 `QUARANTINE`, 140 `EXCLUDE_AUTO`; tổng 31.928,
  clean-core 22.391, review queue 9.537, 2.367 transformations và 194 template
  families.
- **Validation đã chạy:** Validator thường trả `VALID`; xác nhận 25 checksum
  entries, 31.928 parent records, bốn partition kín, 140 duplicate aliases,
  22.391 annotation index rows và mọi ô nhãn còn blank.
- **Rebuild-check:** Đạt `BYTE_REPRODUCIBLE` trên 25 checksum entry với
  `built_at` được tái sử dụng.
- **Unseen semantic audit:** Tập 200 dòng mới có 179 strict ABSA-usable, 16
  pure catalogue, 2 hard non-review và 3 borderline; Wilson 95% của 179/200 là
  84,48--93,03%. Kết quả này được dùng để xây rule 2.1.2 nên trở thành
  calibration evidence.
- **Quyết định:** Không phát hành 2.1.1 cho annotator; giữ snapshot để tái lập
  lịch sử và dựng release 2.1.2.
- **Giới hạn:** Đây là agent-QC một analyst, không phải human gold audit.
- **Phụ thuộc kế tiếp:** Structural catalogue rule 2.1.2, holdout mới,
  double-blind human curation và pilot annotation.

## TASK-20260725-011 — Xuất hồ sơ DOCX và đóng gói bằng chứng

- **Trạng thái:** Đã hoàn tất bản trung gian 2.1.1; bản đó bị thay thế khi
  2.1.2 được chốt.
- **Mục tiêu:** Đồng bộ living protocol với release 2.1.1, tạo DOCX đọc được
  và ghi lại kiểm tra tại mốc lịch sử.
- **Đầu vào:** Protocol Markdown, manifest/checksum của release
  `a1efc1c16f0e8f02`, log builder/validator và kết quả QC.
- **Đã thực hiện:** Cập nhật release ID, phân hoạch, rating/transport,
  template/language/ad/reward rules, lịch sử snapshot bị thay thế và ranh giới
  giữa pre-annotation với gold dataset; render DOCX 38.092 byte và đọc ngược
  bằng Pandoc thành công (798 dòng, 7.541 từ, 45.214 ký tự).
- **Quyết định:** Bản DOCX 2.1.1 chỉ là checkpoint, không được bàn giao như
  tài liệu hiện hành sau khi 2.1.2 tồn tại.
- **Phụ thuộc kế tiếp:** Cập nhật mọi số liệu 2.1.2 và render lại.

## TASK-20260725-012 — Rule 2.1.2, build release và technical closure

- **Trạng thái:** Đã thực hiện và xác thực kỹ thuật.
- **Mục tiêu:** Giảm pure catalogue/non-review còn lọt ở core 2.1.1 nhưng giữ
  mọi quyết định heuristic ở quarantine, không xóa vật lý.
- **Đầu vào:** Parent 31.928 dòng; config/rule 2.1.1; 200-row unseen audit của
  2.1.1; các regression/QC case đã version.
- **Code/config:** Thêm `MODULAR_TITLE_LIST`,
  `EXPANDED_RECURRENT_SPEC_LIST`, `GLUED_CATALOGUE_CLAUSE`; reviewer veto giữ
  dấu; seller/listing/no-evaluation patterns; terminal zero-vowel junk detector
  có DF/metadata/residual-quality guard. Cập nhật builder, validator, config và
  regression tests.
- **Calibration:** Ba structural rule mới có union 8.324 trên parent; 2.292
  hit chỉ do nhánh mới. Positive panel 120/120 là pure/mixed catalogue;
  negative-control 100 giữ được 42/42 natural/mixed nhưng cũng cố ý bỏ lọt 58
  invalid để ưu tiên precision. Hai panel do một analyst/agent đánh giá.
- **Đầu ra:** Release
  `lazada_vi_absa_curation_v2_1_2_20260725`, ID
  `lazada-vi-absa-curation-e42d6c5319faedd3`; 20.429 `KEEP`, 193
  `KEEP_CLEANED`, 11.166 `QUARANTINE`, 140 `EXCLUDE_AUTO`; clean-core 20.622,
  review queue 11.306, 2.377 transformations.
- **Technical validation:** Validator `VALID`; checksum 25/25 release entry,
  49/49 parent, 1.573/1.573 source entry và 6/6 code provenance;
  rebuild-check `BYTE_REPRODUCIBLE`; 74/74 test discoverable trong active
  suite và `py_compile` đạt.
  `git diff --check` chỉ bao phủ tracked diff; code inventory chưa khóa đủ hai
  test file và worktree chưa có commit/tag.
- **Transformation audit:** Replay đủ 10 terminal trim mới; không thấy false
  cleanup. Tám dòng vẫn quarantine, hai residual review vào `KEEP_CLEANED`.
- **Quyết định:** Đóng băng 2.1.2 làm audited pre-annotation release hiện
  hành; không sửa tiếp theo exact ID trước khi có audit/human gate mới.
- **Giới hạn:** Technical validity không đồng nghĩa content validity hay gold
  label validity.
- **Phụ thuộc kế tiếp:** Unseen semantic holdout, delta audit và human
  curation.

## TASK-20260725-013 — Semantic holdout và delta audit 2.1.2

- **Trạng thái:** Đã thực hiện read-only; chưa phải human gold audit.
- **Mục tiêu:** Đo contamination còn lại trong core mới và false-positive
  risk của 1.769 dòng vừa chuyển sang quarantine.
- **Unseen holdout:** Stable sample 250 dòng, SHA-256
  `486b9591ca0fa37cc7720a14d18b4ca2a9d3bae49b5790662ee291e35b935e5c`;
  230 strict usable, 11 pure catalogue, 3 hard non-review, 2
  foreign/gibberish và 4 borderline. Wilson 95% strict usable
  87,97--94,76%; 23/230 usable còn mixed-noisy.
- **Delta audit:** Tất cả 1.769 dòng rời core 2.1.1 đều vào quarantine; sample
  200 có 156 pure catalogue, 37 mixed catalogue+review, 5 borderline và 2
  natural-clean. Analyst xếp 193/200 là operationally quarantine-appropriate;
  natural-clean là 2/200.
- **Ledger:** Persist đủ 250 semantic-holdout row và 200 delta-audit row tại
  `docs/audits/`, với stable hashes; cả hai chỉ lưu text SHA-256 chứ không
  nhân bản review nguyên văn.
- **Quyết định:** Không nới global threshold hoặc thêm blanket buyer/UI veto;
  hai false positive sạch đi human override. Không dùng 20 lỗi holdout để
  overfit 2.1.2; nếu sửa thành 2.1.3 phải lấy holdout khác.
- **Giới hạn:** Nội dung do một analyst/agent phân loại; một số ID panel
  structural cũ không được persist nên không chứng minh được overlap bằng 0
  với mọi calibration panel lịch sử.
- **Phụ thuộc kế tiếp:** Human review của audit/queue, pilot double-blind, IAA
  và adjudication.

## TASK-20260725-014 — Đồng bộ protocol và DOCX hiện hành

- **Trạng thái:** Đã thực hiện và xác thực.
- **Mục tiêu:** Thay bản DOCX trung gian bằng hồ sơ 2.1.2 có collection
  closure, cleaning flow, audit tốt/xấu, reproducibility, bias, privacy và
  ranh giới khẳng định để dùng làm nguồn viết paper.
- **Đầu vào:** Manifest 2.1.2, technical audit, unseen holdout, delta audit,
  protocol Markdown và script Pandoc.
- **Đã thực hiện:** Cập nhật release ID, partition, structural rule, 2.377
  transformation, 80 QC ID, 74 test, collection closure, sampling bias,
  product concentration, semantic/delta audit, machine-readable ledger,
  cookie/profile risk, legal/privacy caveat và external-audit checksum list.
- **Validation đã chạy:** Script Pandoc tạo OOXML hợp lệ và tự đọc ngược hơn
  1.000 ký tự; explicit Pandoc DOCX→plain trả exit 0. Plain text chứa đúng
  release ID 2.1.2, TASK-014 và cảnh báo `src/cookies.txt`.
- **Đầu ra:** `docs/DATASET_CONSTRUCTION_PROTOCOL.docx` hiện hành; kích thước,
  hash, dòng/từ/ký tự của lần render bàn giao được báo cùng artifact.
- **Quyết định:** Task hoàn tất ở tầng hồ sơ kỹ thuật; dependency kế tiếp là
  human curation/pilot chứ không phải thêm heuristic để đuổi theo số lượng.

## TASK-20260725-015 — Đóng inventory Old/Augmented Dataset

- **Trạng thái:** Đã thực hiện read-only; benchmark release riêng chưa tạo.
- **Mục tiêu:** Kiểm lại các count lịch sử, định nghĩa “unique”, mức overlap
  và khóa SHA-256 của 16 XLSX trước khi đề xuất dùng làm benchmark/augmentation.
- **Đầu vào:** 10 file `legacy/data/old_dataset/*.xlsx` và 6 file
  `legacy/data/augmented_dataset/*.xlsx`.
- **Phương pháp:** `openpyxl` read-only/data-only trên `Sheet1`; bỏ
  `reviewContent` rỗng; so exact `str(value)` và diagnostic key
  NFKC+casefold+collapse-whitespace; hash nguyên bytes mỗi file. Môi trường:
  system Python 3.11.4, `openpyxl` 3.1.5.
- **Kết quả:** Old 10.105 nonempty/9.773 exact-unique; Augmented
  11.202/10.877; 9.945 augmented row match Old; exact unique intersection
  9.620; union 11.030; normalized union 11.029; 1.257 dòng
  `llm_synthetic`.
- **Artifact:** `docs/audits/LEGACY_DATASET_AUDIT.md` và
  `docs/audits/legacy_xlsx_sha256.csv`; inventory CSV SHA-256
  `fd7cfcd7ec7fc139017ee265f113482e71ad0c15520ef1570de708670d18396a`.
- **Quyết định:** Không coi Old và Augmented là hai benchmark độc lập;
  synthetic chỉ train-only sau split, có ablation.
- **Giới hạn:** Chưa audit compatibility nhãn theo review--aspect, chưa có
  label validator/versioned benchmark manifest và chưa khóa bằng commit/tag.
- **Phụ thuộc kế tiếp:** Benchmark release riêng và leakage audit với corpus
  mới trước modeling.

## TASK-20260725-016 — Pipeline LLM pseudo-label fail-closed và preflight hai lượt

- **Trạng thái:** Đã triển khai, kiểm thử và chạy preflight kỹ thuật 5 review;
  chưa đạt semantic quality gate và chưa chạy 450/20.622 review.
- **Mục tiêu:** Kiểm tra liệu có thể dùng LLM theo
  `ABSA_ANNOTATION_GUIDELINE_V2` để tạo nhãn phủ rộng mà vẫn giữ đúng ranh giới
  khoa học: nhãn máy là `LLM_PSEUDO_LABEL`, không phải human gold; mọi output
  phải có evidence, uncertainty, provenance, checkpoint và validator
  fail-closed.
- **Đầu vào đã khóa:** Release 2.1.2 ID
  `lazada-vi-absa-curation-e42d6c5319faedd3`; 20.622
  `curated_review_text`; 450 pilot candidate chưa nhãn; Guideline V2
  `2.0.0`, SHA-256
  `58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9`;
  schema LLM `absa-llm-annotation/1.0.0`.
- **Thiết kế blind đã thực hiện:** Model chỉ nhận ba field
  `schema_version`, `blind_id`, `reviewContent`. Rating, category, product/shop
  ID, URL/query/SKU, transport, crawl date, quality/cleaning flag, raw text
  trước cleaning, label cũ và mọi metadata khác chỉ nằm trong private index.
  Text gửi model luôn là `curated_review_text`; 193 dòng `KEEP_CLEANED` không
  bị thay ngược bằng raw text.
- **Schema/quy tắc validator:** Đủ đúng chín aspect và đúng thứ tự; chỉ nhận
  `-1`, `0`, `1`, `2`, chuỗi canonical `"1, -1"` hoặc toàn `null` khi
  `REJECT_NON_REVIEW`. LLM trả exact quote cùng occurrence; code tự replay
  Unicode code-point offset `[start,end)`. Duplicate JSON key, NaN/Infinity,
  key thừa/thiếu, label sai type, quote không tồn tại, occurrence sai,
  evidence/polarity không hỗ trợ label, `OTHER` thiếu note và status/uncertainty
  sai đều bị chặn; không fuzzy-repair hoặc ép lỗi thành neutral/absent.
- **Code/artifact mới:** `src/lazada_collector/llm_annotation.py`,
  `src/lazada_collector/llm_backends.py`,
  `scripts/prepare_llm_annotation_pilot.py`,
  `scripts/run_llm_annotation.py`,
  `scripts/evaluate_llm_annotation_pilot.py`,
  `configs/llm_annotation_v1.json`,
  `configs/llm_annotation_output_schema_v1.json` và
  `tests/test_llm_annotation.py`. Pilot input bất biến nằm dưới
  `data/annotations/llm_pilot_v1_1_20260725`, ID
  `llm-absa-pilot-9df58e1e0efd98ff`; thư mục generated này bị gitignore.
- **Tính toàn vẹn và resume:** Pilot builder kiểm manifest/SHA256SUMS của
  clean-core, pilot CSV và legacy schema; runner kiểm bijection
  `blind_id↔sample_id`, text hash, private-index schema, source release ID/hash,
  pinned guideline/schema SHA và hash code. Mỗi provider attempt được ghi
  atomic, append-only theo số thứ tự; resume replay raw response/evidence,
  không ghi đè hoặc gọi lại attempt hợp lệ. Kiểm thử bao phủ crash sau durable
  attempt, retry failure, checksum bị sửa, path traversal, HTTPS redirect,
  remote HTTP, 401 fatal circuit, 429 pause và completed-resume byte-preserving
  no-op.
- **Backend thực sự đã dùng:** NVIDIA OpenAI-compatible endpoint với
  `mistralai/mistral-medium-3.5-128b`, `temperature=0`, hai pass có prompt
  order/seed riêng. API key chỉ đọc từ biến môi trường `NVIDIA_API_KEY`; giá
  trị key/cookie không được đọc vào artifact hoặc log. Review text đã được
  truyền cho provider bên thứ ba; điều khoản/quota trial phải được kiểm tra
  trước production.
- **Preflight đã chạy:** Năm review đầu được gán ở `pass_a` và `pass_b`.
  Cả 10 terminal record hợp lệ; pass B có một response schema-invalid
  (`Negative label/evidence mismatch`) được validator chặn và lần retry kế
  tiếp hợp lệ. Evaluator xác nhận 45/45 aspect-cell label agreement, 5/5
  full-vector agreement và 5/5 exact-semantic repeat agreement; sinh năm
  consensus pseudo-label và đưa bốn review có negative/neutral vào human queue.
  Đây chỉ là repeat-consistency của cùng một model, không phải IAA và không
  phải accuracy.
- **Usage/độ trễ đo được:** 11 response attempt dùng tổng 106.216 prompt token,
  5.599 completion token, 111.815 token; 10 terminal success dùng 101.674
  token. Pass A có mean 68,91 giây/attempt (26,08--111,03); pass B có mean
  57,99 giây/attempt (13,90--126,92). Full guideline tạo khoảng 9,6k prompt
  token cho mỗi review. Chạy tuần tự 20.622 review × hai pass theo cấu hình này
  vì vậy không khả thi về thời gian/quota; phải batch/amortize prompt, dùng
  concurrency có giới hạn và đo lại trước production.
- **Sự cố được giữ làm audit:** Một run B bị network sandbox trả
  `WinError 10013`; child process bị terminate vẫn ghi muộn sau checksum, nên
  validator phát hiện checksum mismatch. Run này bị loại khỏi evaluator,
  không sửa/reseal. Một run B sạch mới được chạy tới `COMPLETED` và là nguồn
  duy nhất của báo cáo preflight.
- **Kết quả kiểm thử:** `py_compile` đạt; active discovery 138/138 test đạt.
  Evaluator tạo artifact ID
  `llm-pilot-evaluation-558ee0af749ff5f670a8`, trạng thái
  `TECHNICAL_VALIDATION_PASS`, nhưng semantic gate được ghi bắt buộc là
  `NOT_EVALUATED_NO_HUMAN_GOLD`.
- **Quyết định:** Không scale 450 hay 20.622 chỉ vì agreement preflight là
  100%. Trước hết phải khóa prompt sau calibration, tạo 200 review human
  reference double-blind + expert adjudication, đo mention macro-F1, full
  five-state macro-F1, exact cell agreement, mixed/non-review và evidence
  grounding theo gate V2. Nhãn LLM chỉ được dùng train-only/weak supervision;
  dev/test phải human-adjudicated và group-disjoint.
- **Phụ thuộc kế tiếp:** Human annotation/adjudication cho locked reference
  200; sau đó tối ưu batch/concurrency, chạy pilot 450, đánh giá lại và chỉ
  scale nếu semantic gate đạt.

## TASK-20260726-017 — Khóa human-reference 200 và dựng UI double-blind

- **Trạng thái:** Đã thực hiện selection, đóng gói input A/B, triển khai UI,
  strict validator và kiểm thử kỹ thuật. **Chưa có bất kỳ human label nào**;
  artifact hiện tại là `LOCKED_INPUT_PENDING_TWO_HUMAN_ANNOTATIONS`, chưa
  phải human gold, chưa có IAA và chưa adjudicate.
- **Mục tiêu:** Tạo nhanh một workbench ngoài `src/` để hai annotator gán độc
  lập cùng 200 review theo Guideline V2, đồng thời khóa trước sampling design,
  blindness, evidence, provenance và leakage policy để kết quả sau này có thể
  dùng làm semantic quality gate cho LLM.

### Kế hoạch đã đăng ký trước khi chạy

- Không lấy toàn bộ 200 từ pilot 450 vì frame này đã làm phẳng rating
  (`1/2/3/4/5 = 86/86/91/88/99`) và không đại diện cho clean-core
  (`431/146/256/584/19.205`).
- Dùng thiết kế hybrid, báo metric hai panel riêng:
  150 mẫu representative từ clean-core và 50 mẫu challenge từ pilot.
- A và B phải gán đủ cùng 200 source review nhưng nhận permutation và opaque
  ID riêng; không thấy rating, category, product/shop ID, URL/query, nguồn
  crawler, cleaning flag, panel/bin, nhãn cũ, nhãn người khác hoặc output LLM.
- Nếu 200 mẫu được dùng để đánh giá model cuối, phải reserve toàn bộ leakage
  group liên quan khỏi train. Nếu chỉ dùng làm LLM quality gate, vẫn không
  được gộp representative và challenge thành một population estimate tùy ý.

### Đầu vào và khóa nguồn đã thực thi

- Release 2.1.2 ID
  `lazada-vi-absa-curation-e42d6c5319faedd3`, clean-core 20.622;
  `clean_core.jsonl` SHA-256
  `8ade09657deeacf5290eeed090bd0c2f350bd62d923782db38e9912a108e842e`.
- Pilot 450 ID `llm-absa-pilot-9df58e1e0efd98ff`; private index SHA-256
  `a2cbbee4d036c305e136bd79dc9a92f4b045037f5d5d800676fe98979dbf95e1`.
- Semantic holdout 250 SHA-256
  `d54196204393a740b5f0d2acc023d02b6f759c6ab572383a9e49dfb038a72b57`.
- Guideline V2 `2.0.0` SHA-256
  `58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9`.
- Không đọc, sửa hoặc ghi lại `data/raw/`; không dùng rating/sentiment làm
  human label và không dùng 200 review này làm practice.

### Sampling và leakage grouping đã chạy

- Leakage component được dựng bằng union-find trên cùng `product_id`, exact
  canonical-text/duplicate representative, edge trong
  `near_duplicate_candidates` và member của template family. Blank product ID
  không bị nối với nhau. Group ID là hash ổn định của member set.
- **Representative 150:** loại exact row của toàn bộ pilot 450, semantic
  holdout 250 và text hash không bijective; không loại slate
  `curation_audit_1000` vì slate này còn blank, chưa chứa human/model label.
  Hamilton allocation theo rating cho quota
  `1:3, 2:1, 3:2, 4:4, 5:140`, sau đó phân tiếp theo
  category × transport. Stable rank dùng salt
  `human-reference-200-v1-representative-row`; tối đa hai row mỗi leakage
  component.
- Representative thực tế có 131 leakage/product component; category
  `<blank>:1`, beauty 21, electronics 24, fashion 23, food 26,
  home-appliances 18, home-living 14, mother-baby 11, sports 12; transport
  requests-cookie 136 và DOM 14. Không overlap exact ID với pilot hoặc
  semantic holdout.
- **Challenge 50:** frame pilot sau khi loại rank 1--5 đã chạy LLM preflight,
  semantic-holdout ID và mọi group đã vào representative. Mỗi challenge row
  thuộc group riêng, category cap 10 và được chọn round-robin bằng stable
  hash. Các quota ưu tiên thực tế đều đóng đủ:
  `WARRANTY_RETURN_CUE:5`, `AUTHENTICITY_CUE:5`,
  `SHIPPING_PACKAGING_BOUNDARY:5`, `SHOP_SERVICE_CUE:5`, `CONTRAST:8`,
  `NEGATION:5`, `LOW_RATING:5`, `MID_RATING:4`,
  `CLEANED_OR_LONG:4`, `MINORITY_METADATA:2`, `REMAINDER:2`.
  Ngưỡng dài P95 đo trên frame là 293 Unicode character.
- Challenge thực tế có 50/50 component, category lớn nhất 8; rating
  `1:12, 2:13, 3:9, 4:8, 5:8`; transport requests-cookie 45 và DOM 5.
  Các bin trên chỉ là **coverage proxy từ text/metadata**, chưa phải aspect
  mention hay polarity đã được human xác nhận.
- Hai panel không trùng sample hoặc leakage group. Nếu reserve cho model
  evaluation, ledger giữ 6.646/20.622 core row và còn 13.976 row; đây là
  trade-off lớn cần quyết định trước split/training, không được áp dụng ngầm.

### Artifact A/B và provenance đã tạo

- Package:
  `data/annotations/human_reference_v1_20260726`; reference ID
  `human-absa-reference-d849727ea48b3215`; built-at
  `2026-07-26T07:18:01.631564+00:00`.
- Assignment A và B đều có đúng 200 review, cùng tập text SHA-256, nhưng order
  khác nhau và opaque ID không giao nhau. Mỗi public record chỉ có đúng
  `annotation_id`, `reviewContent`, `review_text_sha256`.
- Assignment A SHA-256
  `b9eb94ad79d3eaae071bb71db73cd8aeba23d755f6f2dc52bbb7723818ab06e5`;
  assignment B SHA-256
  `7229c49def115282dac3c8150a4fb8d561da7521f8166339f1646a78042c8a88`.
- Private artifact gồm `crosswalk.jsonl` 200 row,
  `selection_ledger.jsonl` 20.622 row và `group_reservations.jsonl` 6.646
  row. Package tự mang bản sao code selection/common cùng runtime metadata;
  manifest SHA-256
  `fd812548b142ec39accee4ad8ae1213deefce916495d752a6ebe7f5e4e955392`,
  `SHA256SUMS.txt` SHA-256
  `93ad5af23fc3aadcc89e637394d73b8e140ad9ddd38ca0bdbade5e84f8bbb7ae`.
- Rebuild bằng đúng `built_at` trên một output root khác đạt
  `BYTE_REPRODUCIBLE` cho 11/11 file; temp rebuild được xóa sau so hash.

### UI và validator đã triển khai ngoài `src`

- Source mới nằm hoàn toàn tại `human_annotation_ui/`: HTML/CSS/JavaScript
  thuần, Python standard-library localhost server, selection builder, strict
  final-export validator, README và PowerShell launcher. Không thêm npm,
  framework, CDN, analytics, service worker hay runtime dependency mới.
- Root `README.md` đã được thêm mục current-step và lệnh launcher A/B để không
  nhầm workflow reference 200 với các primary batch của frozen corpus cũ.
- Server chỉ bind `127.0.0.1`, chỉ phục vụ một role-specific assignment,
  validate checksum trước startup, đặt CSP/no-store/nosniff/frame-deny,
  không CORS, chặn Host lạ để giảm DNS-rebinding và từ chối mọi POST.
- UI có đúng chín aspect theo thứ tự canonical, năm label không prefill,
  status `LABELED/ESCALATE/REJECT_NON_REVIEW`, uncertainty, notes, exact
  evidence selection, Unicode code-point offset, keyboard shortcuts,
  review navigator/filter, revisit, autosave atomic bằng IndexedDB,
  crash/reload resume, versioned revision ledger, DRAFT backup/import và
  checksum. FINAL bị chặn nếu chưa đủ toàn bộ assignment hoặc còn lỗi.
- Export không chứa review text hay metadata; Python validator join lại với
  frozen assignment, replay evidence occurrence/offset và dùng
  `validate_and_normalize_annotation` làm semantic source of truth. Output
  publish là thư mục mới, không ghi đè.
- Bulk-fill blank thành label `2` chỉ mở sau khi annotator xác nhận đã đọc,
  cần confirm riêng và ghi audit event; blank không bao giờ tự biến thành
  absent.

### Validation và số đo đã thực thi

- Active Python discovery đạt **149/149 test**; riêng human tool có 11 test
  về public blinding, checksum/tamper, A/B role, group cap/disjointness,
  Hamilton allocation, emoji offset, final normalization, DNS-rebinding và
  method refusal.
- Node test runner đạt **8/8 test** cho canonical JSON, Unicode code-point,
  blank-vs-absent, mentioned evidence, mixed, uncertainty và non-review.
- `py_compile`, `node --check`, PowerShell parser cho launcher và
  `git diff --check` đều đạt; `git diff --check` vẫn chỉ phản ánh tracked
  diff trong worktree lịch sử đang bẩn.
- Selenium headless Chrome 150 smoke trên fixture hai review đạt: HTML/script
  trong review chỉ hiển thị như text; đủ chín row; bulk-fill/complete hoạt
  động; IndexedDB khôi phục đúng `1/2` và current review sau reload; browser
  console không có lỗi; mọi resource host là `127.0.0.1`; mobile
  390×844 có horizontal overflow 0 px. Screenshot SHA-256
  `6e72a2bfd6c50adbe2b01b7b000e3a1c82ac3974a9c7cecedbcf33bddd505c71`.

### Quyết định, giới hạn và phụ thuộc kế tiếp

- **Quyết định:** Khóa nguyên package/reference ID hiện tại trước annotation.
  Không chạy LLM 450/20.622 chỉ vì UI hoặc technical validation đạt. Báo
  representative/challenge riêng; chỉ gọi bộ này là human reference/gold sau
  hai lượt human độc lập, strict validation và expert adjudication.
- **Giới hạn:** Chrome smoke dùng fixture hai row chứ không tạo nhãn cho 200
  thật. Local software không thể chứng minh A và B là hai người độc lập nếu họ
  chia sẻ máy/profile/file. IndexedDB/checksum cung cấp durability/integrity,
  không phải encryption hoặc chữ ký mật mã. Challenge cue không bảo đảm aspect
  thật. Chưa có qualification set, hidden expert control, IAA, adjudication
  ledger hoặc semantic accuracy của LLM.
- **Phụ thuộc kế tiếp đã lên kế hoạch nhưng chưa thực hiện:** Annotator A và
  B dùng profile/máy riêng để gán đủ 200; sao lưu định kỳ; curator validate hai
  file FINAL; tính IAA trên raw A/B trước adjudication; expert adjudicate mọi
  disagreement/uncertainty/non-review/mixed và audit ngẫu nhiên nhóm đồng
  thuận. Sau đó mới chạy LLM trên đúng reference, báo mention/polarity/full
  five-state/evidence metric theo panel và quyết định có scale pilot hay
  không.

## TASK-20260726-018 — Pre-annotate 200 theo V2 và dựng luồng human-check tách biệt

- **Trạng thái:** Đã xác thực 10 nhãn calibration, tạo AI pre-annotation đủ
  200, cross-audit, reconcile, đóng gói versioned và smoke-test UI
  human-check. Artifact hiện có trạng thái
  `AI_PREANNOTATION_PENDING_HUMAN_VERIFICATION`; **chưa phải human gold**,
  chưa phải hai lượt human độc lập, chưa có IAA và chưa hoàn tất human-check.
- **Mục tiêu:** Theo yêu cầu mới của người dùng, dùng 10 review đã gán trong
  UI làm tư liệu hiệu chuẩn cùng Guideline V2, tạo gợi ý nhãn/evidence cho đủ
  200 review, rồi cho người dùng kiểm tra và sửa toàn bộ 200. Luồng này phải
  tách khỏi assignment A/B double-blind của TASK-017 để không ghi đè nhãn
  người dùng hoặc mô tả sai provenance.

### Đầu vào và calibration đã thực thi

- Input text vẫn là assignment A đã khóa
  `hra-a-974b86c75d3cc5a4`, 200 review, reference ID
  `human-absa-reference-d849727ea48b3215`; chỉ dùng `reviewContent` và opaque
  ID/hash, không dùng rating, category, product/shop ID, crawler metadata hoặc
  nhãn cũ.
- Guideline là `ABSA-ANNOTATION-GUIDELINE-V2` phiên bản `2.0.0`, SHA-256
  `58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9`.
- Human DRAFT người dùng cung cấp:
  `hra-a-974b86c75d3cc5a4-draft-2026-07-26T08-54-12-069Z.json`, file SHA-256
  `c0ba37377793091af9dd19a57fa92bbfd076cb1baf2ec5748c39d9146aff0c8c`,
  payload SHA-256
  `232934b4b599649d402c6325fd2b1b8da106e506695520b0819d6abab82b6682`.
  Strict validator xác nhận đúng schema/bijection/hash: 200 record, 10
  completed-valid, 190 incomplete; 9 `LABELED` và 1 `REJECT_NON_REVIEW`.
  Bản DRAFT nguồn được giữ nguyên, không bị AI ghi đè.
- Mười mẫu được dùng để hiểu cách người dùng áp guideline, nhưng Guideline V2
  vẫn là thẩm quyền cao nhất khi có xung đột. Các ranh giới được audit rõ gồm:
  mismatch brand/image không tự suy fake; quà tặng thuộc
  `Giá cả & Khuyến mãi`; chuỗi quảng bá không có trải nghiệm đi cleaning;
  fulfillment delay có thể thuộc `Vận chuyển`; câu thuật lại phản hồi shop
  không tự động là negative.
- Sau reconciliation, so sánh diagnostic trên 10 calibration record đạt
  status exact 9/10, full-vector exact 6/10 và aspect-cell exact 85/90
  (`94,44%`). Bất đồng còn ở status mẫu 2; Authenticity mẫu 3--4;
  Price/ShopService mẫu 5; Shipping mẫu 8. Đây **không phải IAA hoặc
  accuracy**: chỉ có một human draft, n=10 rất nhỏ và AI đã nhìn các mẫu này
  trong calibration.

### Gán nhãn, cross-audit và reconciliation đã thực thi

- 200 record được chia theo assignment position thành ba batch `1--67`,
  `68--134`, `135--200` và được gán bằng Codex theo Guideline V2; không gọi
  provider/API LLM ngoài. Mỗi record có đúng chín aspect, status, exact
  evidence quote + one-based occurrence, uncertainty và note khi cần.
- Validator hiện hữu replay mọi quote trên canonical `reviewContent`, tính
  Unicode code-point offset `[start,end)`, kiểm label/evidence polarity,
  mixed positive+negative, neutral-vs-absent, status/uncertainty và reject
  toàn-null. Không fuzzy-repair, không tự đổi blank thành `2`.
- Batch 1--67 và 135--200 được một agent khác đọc lại toàn bộ; batch 68--134
  được agent chính đọc lại toàn bộ. Cross-audit tạo 27 issue record:
  11 proposal `CHANGE` từ hai reviewer và 16 mục `CHECK`/root check. Ledger
  quyết định đóng đủ 27/27:
  `APPLY_PROPOSED:10`, `APPLY_CUSTOM:4`,
  `RETAIN_ORIGINAL:6`, `DEFER_HUMAN_CHECK:7`.
- Reconciliation tạo batch mới, không sửa ba raw annotation batch. Có 14
  record thay đổi tại vị trí
  `4, 6, 8, 20, 33, 35, 43, 46, 141, 158, 178, 184, 191, 197`.
  Các sửa chính: bỏ suy diễn authenticity từ Lazada Mall/“hàng chuẩn”; trả
  chuỗi quảng bá về non-review; bổ sung lỗi sản phẩm/ShopService bị bỏ sót;
  sửa Quality--Performance cho cảm quan; bỏ mixed giả do thiếu phụ kiện;
  không dùng boilerplate làm evidence; thu hẹp evidence vượt phạm vi aspect.
  `reconciliation_report.json` SHA-256
  `1cfc698b37c56cffdf7873f5742b3fbcf5d674cbc39230106515c9c1d2b02949`.
- Tất cả raw batch, cross-audit, decision ledger, reconciled batch, builder,
  reconciliation script và independent package validator được giữ trong
  provenance/checksum; không đọc, sửa hoặc xóa `data/raw/`.

### Artifact và phân bố pre-annotation đã tạo

- Package versioned:
  `data/annotations/human_reference_ai_preannotation_v1_20260726`, built-at
  cố định `2026-07-26T09:19:07.902462+00:00`; manifest SHA-256
  `89284c3d9bebe1a91443371c0280a958e8c8ed186561851213a4719c5bfa620b`.
- `ai_preannotations.jsonl` có đúng 200 record, SHA-256
  `d0fe6cf857f989985c0b43106ae3f61b266f295283d7394cb1c0809f387c69c0`.
  Trạng thái: 180 `LABELED`, 13 `ESCALATE`, 7
  `REJECT_NON_REVIEW`; 20 record có uncertainty/non-review và không có
  `LABELED` record nào toàn chín nhãn `2`.
- Có 499 mentioned aspect-cell; 32 neutral cell; 27 mixed cell; 36 review có
  cả positive và negative ở cấp review. Neutral theo aspect:
  Quality 9, Performance 8, Description 1, Price 7, Shipping 0,
  Packaging 0, ShopService 1, Warranty/Return 2, Authenticity 4. Mixed theo
  aspect: Quality 10, Performance 12, Description 2, Price 0, Shipping 1,
  Packaging 1, ShopService 0, Warranty/Return 0, Authenticity 1. Đây là phân
  bố AI trước human-check, không phải gold prevalence và không bị ép quota.
- Human-review assignment riêng:
  `hra-ai-review-033a7c9c9ab4eda6`, file SHA-256
  `9c04d785f7c47e5b6d24d3f0703ecb87096b8b3e9aeace95bb71e0bde430a4f2`.
  Suggestion set `ai-suggestions-bd87f57e47475c49`, file SHA-256
  `80cd03b02ae85384ed3b8ae09cedc2b8c9fcb3d3482472c80e1b7d8b6faffa70`.
- Package có 29 file; top-level `SHA256SUMS.txt` kiểm kê 28 file còn lại và
  có SHA-256
  `d80e3dabfe8dc5f5fe423d8cf5f0a40a4f83bc8b9b9546ec84dd759a500fd7ca`.
  Rebuild độc lập bằng cùng input và `built_at` đạt byte-identical 29/29 file.

### UI human-check đã triển khai và kiểm thử

- `human_annotation_ui.serve` nhận optional `--suggestions`, strict-validate
  suggestion schema/checksum/assignment/guideline/ID/hash/evidence trước khi
  phục vụ, và công bố mode riêng qua `/review-mode.json`. Normal double-blind
  mode không tự tải hoặc nhìn thấy suggestions.
- Trong AI-assisted mode, nhãn/evidence được seed nhưng `annotator_id` vẫn
  rỗng, `read_complete=false`, `complete=false`, revision ledger rỗng. Người
  kiểm tra phải đọc, sửa hoặc xác nhận rồi hoàn tất từng record; audit event
  ghi suggestion-set/hash. Banner cố định cảnh báo đây là gợi ý AI, chưa phải
  human gold.
- Launcher mới:
  `.\human_annotation_ui\start_ai_review.ps1`, mặc định port `8770`. Assignment
  ID riêng làm IndexedDB session tách khỏi role A cũ; autosave, backup/import,
  revision và FINAL validator hiện hữu vẫn được dùng. README/root README ghi
  rõ output chỉ được gọi `AI-assisted, human-verified` sau khi đủ 200/200 và
  strict validation.
- Python active discovery đạt 153/153 test; Node đạt 8/8; `py_compile`,
  `node --check` và `git diff --check` đạt (chỉ còn cảnh báo LF→CRLF lịch sử
  ở `.gitignore`/`README.md`). Independent package validator trả
  `VALID_AI_PREANNOTATION_PACKAGE`, 200 record, 10 calibration record và 28
  checksummed file.
- Headless Chrome smoke trên **gói 200 thật** đạt: banner AI-assisted hiện
  đúng; suggestions/status/label/evidence được seed; progress ban đầu 0/200;
  hoàn tất một record rồi reload khôi phục 1/200; không có severe console
  error; resource host chỉ `127.0.0.1`; mobile 390×844 overflow 0 px.
  Screenshot SHA-256
  `762653a4322ac530487bb7fb15c0af65e7c4fb82635c2a38bbad75ec98d51dca`.

### Quyết định, giới hạn và phụ thuộc kế tiếp

- **Quyết định đã thực thi:** Giữ nguyên A/B double-blind package và human
  DRAFT nguồn; phát hành suggestions bằng assignment/session mới. Giữ
  `ESCALATE`, `REJECT_NON_REVIEW` và cross-audit uncertainty trong UI thay vì
  ép đủ label hoặc tự loại record. Không gọi artifact này là human gold,
  benchmark hoặc final dataset.
- **Giới hạn:** Human reviewer sẽ nhìn AI suggestion nên có anchoring bias;
  FINAL sau bước này là AI-assisted human-verified, không phải independent
  expert gold và không được dùng để ước lượng không thiên lệch performance của
  chính AI đã tạo suggestion. Calibration n=10 không đủ qualification gate.
  13 escalation, 7 non-review và 7 deferred cross-audit case bắt buộc được
  kiểm tra cẩn thận. Các count neutral/mixed hiện tại có thể thay đổi sau
  human review.
- **Đã lên kế hoạch nhưng chưa thực hiện:** Người dùng chạy
  `start_ai_review.ps1`, đọc và confirm/edit đủ 200, sao lưu định kỳ, export
  FINAL; curator chạy `human_annotation_ui.validate_export` với derived
  assignment và xuất versioned validated artifact. Sau đó phải tính delta
  AI-before/human-after, resolution của mọi escalation/reject/deferred và
  phân bố final.
- **Phụ thuộc khoa học tiếp theo:** Nếu paper cần một benchmark/gold Q1 ít
  anchoring, vẫn phải có annotator B hoặc expert gán blind độc lập trên tập
  khóa, tính IAA trước adjudication và adjudicate theo TASK-017. Không được
  thay IAA bằng tỷ lệ người dùng giữ nguyên suggestion.

## TASK-20260727-019 — Tạo, audit và phát hành tranche 5.000 AI pseudo-label theo Guideline V2

- **Trạng thái:** Đã thực thi đủ quy trình chuẩn bị, qualification, sinh nhãn,
  replay/seal, semantic audit, phát hành versioned và independent validation.
  Release có đúng 5.000 record và mang trạng thái
  `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`; **không phải human gold,
  benchmark độc lập hay training label đã được xác nhận**.
- **Mục tiêu:** Dùng các record người dùng đã human-check làm calibration cho
  Guideline V2, tạo tranche đầu tiên gồm 5.000 review từ clean-core, giữ phân
  phối tự nhiên, loại leakage với human-reference, bảo toàn exact evidence và
  tạo chain-of-custody đủ chi tiết để tiếp tục human verification và mô tả
  trung thực trong paper.
- **Phân biệt kế hoạch và thực thi:** Các mục ghi “đã thực thi” bên dưới đã có
  artifact/checksum hoặc log đo được. Human verification 1.978 record, expert
  adjudication, annotator B blind, IAA, training và đánh giá mô hình **mới chỉ
  là kế hoạch**, chưa được thực hiện trong task này.

### Đầu vào và calibration đã thực thi

- Nguồn canonical là release
  `lazada_vi_absa_curation_v2_1_2_20260725`, release ID
  `lazada-vi-absa-curation-e42d6c5319faedd3`. `clean_core.jsonl` có 20.622
  record, SHA-256
  `8ade09657deeacf5290eeed090bd0c2f350bd62d923782db38e9912a108e842e`;
  source manifest SHA-256
  `4a45a3b83f1bebf9d87751a61aaab168b3d957321935403ce06eb4477536128e`.
  Không đọc, sửa hoặc xóa `data/raw/` trong task này.
- Human export đầu vào là DRAFT AI-assisted, file SHA-256
  `daab2866224a02a4ed5280157a444c75fcdc76138ea43b54adaa07827378cd16`,
  payload SHA-256
  `9d5305058180e955c6e1cf0fe7dd2ebef588770c8995e502559bc96a9658c760`.
  Strict validation xác nhận 88/200 record complete-valid và loại 112 record
  chưa hoàn tất. Trong 88 record có 79 `LABELED`, 3 `ESCALATE` và 6
  `REJECT_NON_REVIEW`.
- Đây là nhãn người dùng xác nhận/sửa từ AI suggestion, không phải annotation
  blind. Audit phát hiện nguy cơ anchoring: phần lớn vector vẫn giống seed và
  một số confirmation xung đột Guideline V2. Vì vậy 68 clear case được chấp
  nhận làm calibration; 20 uncertainty/non-review/mixed hoặc semantic-conflict
  case được giữ nguyên trong provenance nhưng mang quyết định
  `EXCLUDE_PENDING_EXPERT_ADJUDICATION`, không được dùng làm prompt precedent
  hoặc diagnostic truth. Calibration payload SHA-256 của 68 case là
  `15b3cc01fe3455f187c427bdf0d35e5eb94d4fda63be10ce2906a3ca0471410c`.
- Guideline authority là `ABSA_ANNOTATION_GUIDELINE_V2.md` phiên bản 2.0.0,
  SHA-256
  `58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9`.
  Mọi output dùng đúng chín aspect và năm label `-1`, `0`, `1`, `2`,
  `"1, -1"`; `REJECT_NON_REVIEW` dùng chín null theo contract.

### Chống leakage và chọn 5.000 record đã thực thi

- Không chỉ loại đúng 200 human-reference row. Toàn bộ 6.646 record thuộc 181
  leakage group trong `private/group_reservations.jsonl` được reserve; ledger
  SHA-256
  `a8ba153b6211f47f6d0f8f525cb62873ff069c47737d585a73949a7fc4d8357b`.
  Frame an toàn còn 13.976 record. Code kiểm join assignment ↔ crosswalk ↔
  reservation ↔ clean-core bằng ID, text hash, source release và reference
  flag trước khi chọn.
- Chọn đúng 5.000 bằng hierarchical Hamilton largest-remainder với số hữu tỉ
  `Fraction`: phân bổ theo rating trước, sau đó category × collection transport
  trong từng rating. Tie-break/rank dùng SHA-256 gắn với source manifest,
  human-reference manifest, reservation ledger, `sample_id` và curated-text
  hash. Ordered membership SHA-256 là
  `4eb4e7a7b082837b47f2fbb7ff627daaee66a8655fa964daa2049d63f5743869`.
- Phân bổ rating thực tế: `1:110`, `2:34`, `3:66`, `4:156`, `5:4634`.
  Category: blank 7, beauty/personal-care 677, electronics 881, fashion 785,
  food/beverage 845, home-appliances 584, home-living 466, mother/baby 314,
  sports/outdoors 441. Transport: requests 3, requests-cookie 4.931 và
  Selenium DOM 66. Curation status: 4.952 `KEEP`, 48 `KEEP_CLEANED`.
  Không áp quota polarity/neutral/mixed.
- Kết quả kiểm: 5.000 ID, text hash và rank đều duy nhất; overlap exact
  human-reference = 0; overlap reserved group = 0. Prepared manifest SHA-256
  `067ed254b6075c7b3d93e8c63d200ec5a5041f41ac92b7e7cf9f846355304726`;
  `INPUT_SHA256SUMS.txt` SHA-256
  `5c7a0a9378289b35a7e5857207340b168923f0e6437d905c0bedec5f9dcc64a1`.

### Qualification model và cấu hình sinh nhãn đã thực thi

- Các thử nghiệm không đạt hoặc không phù hợp đã được dừng thay vì hạ gate:
  NVIDIA `openai/gpt-oss-20b` hoàn tất 20 diagnostic nhưng chỉ đạt
  aspect-cell `158/180 = 0,8778`, full-vector `6/20 = 0,30`, mention F1
  `0,7692`; gate `FAIL`. Ministral-14B trả HTTP 410 đúng ngày end-of-life.
  Llama-3.3-70B/Mistral-Nemotron bị timeout hoặc provider overload.
  Nemotron-3-Nano sinh chain-of-thought/non-JSON hoặc sai envelope khi tắt
  thinking. Mistral Medium 3.5 128B có response hợp lệ nhưng khoảng 154 giây
  cho batch 4, không phù hợp throughput 5.000. Không output nào từ các thử
  nghiệm này được trộn vào tranche chính.
- Backend được chọn là Codex CLI authenticated session, model
  `gpt-5.6-terra`, reasoning effort `medium`, temperature-equivalent
  deterministic instruction, batch size 20. Structured output schema SHA-256
  `0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`;
  compact prompt version `absa-ai-compact-v1.0.0`; system-prompt SHA-256
  `4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`.
- Review là untrusted text. Codex subprocess đã tắt shell, app, browser,
  computer-use, image generation, multi-agent và plugin features; dùng
  environment allowlist để không forward NVIDIA/OpenAI API key. Chỉ opaque
  annotation ID, review text và retrieved human-confirmed examples được đưa
  vào prompt; private source/sample/product metadata ở local.
- Diagnostic cuối dùng cùng batch size 20 và cùng model/prompt/schema với
  primary. Independent recomputation trên 20 holdout clear case đạt:
  status exact `20/20 = 1,00`; full-vector exact `12/20 = 0,60`;
  aspect-cell exact `171/180 = 0,95`; mention precision/recall/F1 đều
  `46/(46+3) = 0,9388`; polarity exact khi cả hai cùng mention
  `43/46 = 0,9348`. Cả năm gate đã khóa trước đều `PASS`.
  `diagnostic_metrics.json` SHA-256
  `333dc2742a8ac995331f6ea0d88ba0d77b4e271868da39c0edf9bfdcb47a35ee`;
  gate SHA-256
  `125c6e2768981d782b091f292044b36c1289d5cdf69308e2fdc6c9895c3408cd`.
  Đây chỉ là prompt-alignment diagnostic với AI-assisted human-confirmed
  holdout, không phải accuracy hay IAA.

### Primary generation, retry và exact-evidence validation đã thực thi

- Primary chạy resumable qua hai invocation: invocation đầu dừng có kiểm soát
  sau checkpoint 180 để tăng từ 4 lên 8 worker; invocation sau tự bỏ qua 180
  record đã ghi atomic và tiếp tục cùng model, batch size, reasoning,
  prompt/schema hash. Khoảng thời gian từ attempt đầu đến attempt cuối là
  2.867,5 giây; runner summary hoàn tất ở
  `2026-07-27T08:38:04.780458+00:00`.
- Có 362 provider attempt: 250 `VALID`, 110 `PARTIAL_VALID` và 2
  `SCHEMA_INVALID`; 0 terminal backend failure. Partial parser chỉ lưu dòng
  hợp lệ và retry đúng dòng lỗi; không chấp nhận quote không phải literal
  substring, occurrence sai, label/evidence polarity sai, absent aspect còn
  evidence, ID/hash sai hoặc envelope thừa/thiếu. Tổng provider wall-time cộng
  dồn qua các worker là 21.141,2 giây, median 71,8 giây, P95 93,8 giây,
  maximum 124,6 giây; đây không phải elapsed wall-clock vì tám worker chạy
  song song. Codex CLI không trả token usage, nên token/cost không được suy
  đoán.
- Quote normalization bị thu hẹp thành **case-only unique match**. Không trim
  prefix/suffix vì có thể làm mất từ polarity như “kém” hoặc “chả”. Toàn bộ
  5.000 record chỉ có 2 record được sửa capitalization case-only; cả hai có
  repair ledger và được đưa vào human-review queue.
- Kết quả đúng 5.000/5.000 record, 0 missing, 0 duplicate ID/hash/rank và 0
  terminal failure. Status: 4.598 `LABELED`, 320 `ESCALATE`, 82
  `REJECT_NON_REVIEW`. Có 12.402 mentioned aspect-cell, 556 mixed
  aspect-cell và 1.004 review có cả positive lẫn negative ở cấp review.

| Aspect | -1 | 0 | 1 | mixed | 2 | null |
|---|---:|---:|---:|---:|---:|---:|
| Chất lượng sản phẩm | 321 | 145 | 2.441 | 209 | 1.802 | 82 |
| Hiệu năng & Trải nghiệm | 276 | 315 | 2.317 | 281 | 1.729 | 82 |
| Đúng mô tả | 335 | 15 | 901 | 33 | 3.634 | 82 |
| Giá cả & Khuyến mãi | 69 | 114 | 1.227 | 16 | 3.492 | 82 |
| Vận chuyển | 110 | 24 | 1.246 | 6 | 3.532 | 82 |
| Đóng gói | 99 | 23 | 876 | 9 | 3.911 | 82 |
| Dịch vụ & Thái độ Shop | 128 | 9 | 571 | 0 | 4.210 | 82 |
| Bảo hành & Đổi trả | 38 | 34 | 39 | 1 | 4.806 | 82 |
| Tính xác thực | 24 | 24 | 125 | 1 | 4.744 | 82 |

### Replay/seal và semantic audit đã thực thi

- Sealer độc lập không tin `run_summary`: nó dựng lại target/example/retry
  message và prompt hash, parse lại từng stored response, đối chiếu annotation
  và repair với đúng `batch_id + valid_attempt`, kiểm model/prompt/reasoning/
  schema/frozen implementation, rồi mới tạo inventory. Diagnostic seal xác
  nhận 20 target từ 1 attempt; primary seal xác nhận 5.000 target từ 362
  attempt; 0 recovered failure marker. Diagnostic run-manifest SHA-256
  `925bff870c9f764e3ea4ba774b45051717746362c8535b36362de4b836bc6b61`;
  primary run-manifest SHA-256
  `fd004e4b865794deada1c772ee71ba3ffd1946e46a6d524310282bd7a6e4568c`.
- Một AI semantic audit riêng được khóa trên snapshot rank 1--1.060 và lấy
  deterministic 60 record: mỗi nhóm REJECT, ESCALATE, neutral, mixed,
  nhiều-aspect và clear-random có 10 record. Cả 60/60 đủ chín aspect và
  188/188 evidence span replay chính xác; 0 repair trong sample.
  Kết luận bảo thủ: 41 không thấy lỗi vật chất, 15 lỗi major và 4
  minor/boundary. Trong clear-random vẫn có 2/10 major; lỗi thường gặp là bỏ
  sót aspect phụ, nhầm Quality ↔ Performance/Description và nhận sai câu
  marketing như review. Audit manifest SHA-256
  `7cbd157cf554fb5fcde3b4256828b40114d0c8340460cbf9a82d1679dedf81d9`.
  Sample cố ý oversample ca khó và reviewer cũng là AI, nên các tỷ lệ này
  **không phải accuracy, IAA hoặc human error rate**. Audit không sửa âm thầm
  model output; issue được giữ trong ledger để human adjudication.

### Release, review queue và validation đã thực thi

- Release self-contained nằm tại
  `data/annotations/absa_ai_tranche_5000_v1_20260727/final/`. Canonical là
  `ai_pseudo_labels.jsonl`, 5.000 row, SHA-256
  `a58da3b437e03221a84845834ca3733af06b2e711d77932b3f77312c940823cc`.
  Compatibility CSV SHA-256
  `5c9b16c250189b32488c9f1be90832e2f65ddbbd549977553e4733eac9303909`;
  decision ledger SHA-256
  `39d82cb17f7ed98bde323461c56b270b53ba6231f92b3aa395a2d229d1146bd4`;
  human queue SHA-256
  `f436db8dd9e9a5e9c3562b23abba5afcfb42b1f835cc1327231b3ae112cb3eb7`.
- Queue có 1.978 unique record. Mandatory flags gồm mọi E/R, uncertainty,
  neutral, mixed, all-aspect-absent, ít nhất năm mentioned aspect,
  normalization repair, marketing-like text, non-empty note, cross-aspect
  reused evidence và overlong evidence. Ngoài ra có SHA-random 10% và explicit
  coverage cho mỗi rating/category/transport và rare aspect-label stratum.
  Số flag chính: `ESCALATE:320`, `REJECT:82`, `NEUTRAL:649`, `MIXED:541`,
  `UNCERTAINTY:402`, `NONEMPTY_NOTES:402`, `MANY_ASPECTS:243`,
  `MARKETING_LIKE:75`, `CROSS_ASPECT_REUSE:44`, `OVERLONG:14`,
  `ALL_ABSENT:11`, `CASE_ONLY_REPAIR:2`. Một record có thể có nhiều flag.
- Final release inventory có 51 artifact, bao gồm guideline/schema/code
  snapshot, calibration, 20 diagnostic record, recomputed metrics/gate,
  diagnostic/primary seals, semantic-audit ledger và finalizer/validator code.
  Không file nào chứa đường dẫn tuyệt đối `C:/Users/...`.
  Final manifest SHA-256
  `bc01e6727b73e5962aa14affea4f3628917554572087a84da951ad69e03f5da1`;
  `SHA256SUMS.txt` SHA-256
  `559356d5884b68ca316e6f212666e8992e817aa10a59a4fd707f51f6f14edf4b`.
- Independent validator tự tính lại gate, kiểm diagnostic-primary config
  equality, seal/checksum closure, exact schema/evidence, source metadata,
  5.000 JSONL/CSV/ledger joins, terminal state và queue. Kết quả:
  `VALID`, 5.000 record, 1.978 queue record, exact-reference overlap 0,
  reserved-group overlap 0, terminal state
  `PUBLISHED_PENDING_HUMAN_VERIFICATION:5000`.
- Validation code đạt 192/192 Python test, gồm forged-PASS, tamper, resume,
  model/schema/source mismatch, missing seal và path portability; annotation
  UI đạt 8/8 JavaScript test. `py_compile` cho preparation/runner/sealer/
  finalizer/validator/audit code đều đạt.

### Quyết định, giới hạn và phụ thuộc tiếp theo

- **Quyết định đã thực thi:** Giữ nguyên toàn bộ AI output và phát hành dưới
  nhãn pseudo-label pending human verification. JSONL là canonical vì giữ
  evidence/uncertainty/provenance; CSV chỉ là projection. Không dùng rating làm
  input gán nhãn; không fit quota polarity; không nhập 20 calibration conflict
  vào prompt; không sửa label theo semantic audit mà không có adjudication
  ledger.
- **Giới hạn khoa học:** 88 human confirmation là DRAFT và thấy AI seed;
  diagnostic holdout chỉ có 20 clear case từ cùng nguồn assisted, nên có
  confirmation bias và không ước lượng generalization. AI semantic audit cho
  thấy lỗi ngữ nghĩa còn đáng kể ngay cả khi schema/evidence hoàn hảo. Rating
  distribution bị chi phối bởi 5 sao theo frame tự nhiên. 1.978 queued record
  chưa được người thật xử lý. Chưa có annotator B blind, IAA, expert
  adjudication, human-label accuracy, downstream model experiment hoặc
  external benchmark evaluation.
- **Giới hạn provenance:** SHA256SUMS chứng minh integrity closure nội bộ,
  không tự chứng minh authenticity nếu một người có thể sửa cả artifact lẫn
  checksum. Manifest digest hiện chưa được neo vào signed Git tag, DOI/Zenodo
  metadata hoặc kho immutable bên ngoài.
- **Kế hoạch chưa thực hiện — phụ thuộc kế tiếp:** (1) dựng/điều chỉnh UI để
  human-review 1.978 queue record, ưu tiên major audit/E/R/uncertainty/mixed/
  marketing; (2) expert adjudicate mọi disagreement và khóa thêm quy tắc
  sensory Quality--Performance cùng câu “tặng shop 5 sao”; (3) xuất release
  version mới, không ghi đè release này, và báo delta AI-before/human-after;
  (4) dùng annotator độc lập blind trên benchmark/reference, tính IAA trước
  adjudication; (5) chỉ sau các bước đó mới quyết định phần nào được dùng làm
  gold, silver/pseudo training data hoặc chỉ làm unlabeled corpus; (6) neo
  final manifest digest vào một external immutable/signed record trước khi
  paper/public release.

## TASK-20260727-020 — Chốt LLM tạo nhãn và sổ trạng thái ĐÃ/ĐANG/SẼ xử lý

- **Trạng thái:** Đã đối chiếu artifact provenance và cập nhật sổ trạng thái
  tổng hợp. Task này chỉ thay đổi tài liệu protocol; không chạy thêm LLM,
  không sửa nhãn, không ghi đè release và không đọc/sửa/xóa `data/raw/`.
- **Mục tiêu:** Trả lời không nhập nhằng model nào đã tạo 5.000 nhãn chính
  thức, đồng thời gom toàn bộ pipeline thành ba nhóm: thủ tục đã thực thi,
  trạng thái hiện tại và thủ tục mới chỉ được lên kế hoạch. Các task trước
  vẫn là nguồn chi tiết; mục này là ảnh chụp trạng thái dùng để theo dõi và
  viết paper.
- **Đầu vào đã đối chiếu:** `runs/primary/run_summary.json`,
  `runs/primary/run_manifest.json`, diagnostic metrics/gate, semantic-audit
  manifest, final `manifest.json`, `SHA256SUMS.txt`, release curation 2.1.2,
  human-reference DRAFT và TASK-001--TASK-019.
- **Phương pháp/thay đổi tài liệu:** Đọc metadata đã đóng dấu thay vì suy từ
  hội thoại; phân biệt model sinh nhãn với model chỉ được thử nghiệm hoặc
  model làm audit; lập inventory theo từng phase và đánh dấu rõ
  `ĐÃ THỰC THI`, `TRẠNG THÁI HIỆN TẠI` hoặc `KẾ HOẠCH CHƯA THỰC THI`.

### LLM nào đã tạo bộ 5.000 nhãn?

- **Model sinh toàn bộ nhãn trong release 5.000:** `gpt-5.6-terra`, gọi qua
  Codex CLI bằng authenticated local session; backend `codex`, logical
  endpoint `codex-cli://local-authenticated-session`, reasoning effort
  `medium`, batch size 20, tối đa 3 retry, timeout 420 giây và 8 worker ở
  invocation hoàn tất. Prompt là `absa-ai-compact-v1.0.0`, SHA-256
  `4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`;
  output schema SHA-256
  `0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`.
- `gpt-5.6-terra` cũng là model chạy diagnostic cuối cùng cùng cấu hình
  prompt/schema trước primary. Diagnostic 20 record đạt đủ gate đã khóa:
  status exact 20/20; full-vector exact 12/20; aspect-cell exact 171/180;
  mention F1 0,9388 và polarity exact 43/46.
- NVIDIA `openai/gpt-oss-20b`, Mistral Medium 3.5 128B, Ministral 14B,
  Llama 3.3 70B, Mistral-Nemotron và Nemotron-3-Nano chỉ là các ứng viên
  screening/diagnostic bị loại do chất lượng, tốc độ, end-of-life, overload
  hoặc sai output contract. **Không có record nào từ các model này được trộn
  vào bộ 5.000.**
- 88 record người dùng đã confirm/edit là human-assisted reference, không
  được gọi là nhãn do `gpt-5.6-terra` tự xác nhận. Semantic audit 60 record là
  lượt kiểm tra bằng AI/Codex và không thay đổi nhãn gốc; nó không phải human
  accuracy hay IAA.

### ĐÃ XỬ LÝ — thủ tục có artifact/log và kết quả đo

| Phase | Việc đã thực thi | Kết quả/đầu ra đo được |
|---|---|---|
| Thu thập và đóng băng nguồn | Crawl bằng các transport đã ghi provenance; kiểm kê manifest, file sản phẩm/review và policy | 31.949 dòng review JSON hợp lệ; 31.928 record canonical thuộc `substantive_vi_v2`; không còn crawler run đang chạy |
| Canonical hóa và làm sạch | Chuẩn hóa text, kiểm duplicate, câu quảng cáo/catalogue/non-review, template, PII và giữ per-record decision ledger; không xóa raw | Release curation 2.1.2 phủ đủ 31.928 record: 20.429 `KEEP`, 193 `KEEP_CLEANED`, 11.166 `QUARANTINE`, 140 `EXCLUDE_AUTO`; clean-core 20.622 |
| Định nghĩa bài toán | Khóa Guideline V2, chín aspect, năm giá trị polarity, exact-evidence và status contract | Guideline 2.0.0 và schema/version/hash đã được đóng gói trong provenance |
| Human-reference và UI | Tạo assignment 200, UI resume/export/validate; người dùng confirm/edit một phần; audit calibration chống anchoring | 88/200 complete-valid: 79 `LABELED`, 3 `ESCALATE`, 6 `REJECT_NON_REVIEW`; 68 clear case dùng calibration, 20 case giữ chờ expert |
| Chống leakage và chọn tranche | Reserve toàn bộ group liên quan human-reference; chọn deterministic theo phân phối tự nhiên rating × category × transport, không fit quota polarity | Reserve 6.646 record thuộc 181 group; safe frame 13.976; chọn đúng 5.000; exact-reference overlap 0; reserved-group overlap 0 |
| Qualification LLM | So sánh các candidate; chỉ scale model qua diagnostic gate với đúng production config | Chọn `gpt-5.6-terra`; diagnostic cuối `PASS`; mọi output của candidate bị loại không đi vào primary |
| Sinh nhãn 5.000 | Chạy resumable, validation fail-closed, retry đúng record lỗi, exact quote/evidence và atomic record write | 5.000/5.000 hoàn tất; 4.598 `LABELED`, 320 `ESCALATE`, 82 `REJECT_NON_REVIEW`; 12.402 aspect được mention; 556 ô mixed; 1.004 review multi-polarity |
| Replay, seal và audit | Parse lại stored responses, kiểm model/prompt/schema/hash; audit semantic deterministic sáu strata | Primary seal đủ 5.000 từ 362 provider attempt; audit 60 record có 188/188 evidence replay đúng, 41 không thấy lỗi vật chất, 15 major, 4 minor/boundary |
| Phát hành và kiểm thử | Xuất versioned JSONL canonical, CSV projection, ledger, review queue, manifest/checksum; chạy validator/test | Release `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION` là `VALID`; queue 1.978 record; 192/192 Python test và 8/8 JavaScript test đạt |
| Tài liệu hóa | Ghi tuần tự quyết định, số đo, giới hạn và next dependency trong protocol; render DOCX | TASK-001--TASK-020 là audit trail của dự án; DOCX được render lại và kiểm tra round-trip bằng Pandoc sau cập nhật này |

### TRẠNG THÁI HIỆN TẠI — đang ở đâu

- Không có tiến trình crawl hoặc LLM labeling nền nào đang tiếp tục. Việc sinh
  tranche 5.000 và phát hành release v1 đã hoàn tất.
- Dự án đang ở **cổng human verification/expert adjudication**. Release v1 bị
  đóng băng ở trạng thái `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`, chưa
  được gọi là gold và chưa nên dùng như ground truth để báo kết quả paper.
- Hàng đợi cần người kiểm có 1.978 record duy nhất. Ưu tiên trước là 15
  major + 4 minor/boundary case trong semantic audit, sau đó E/R,
  uncertainty, neutral, mixed, marketing-like, many-aspect và mẫu ngẫu nhiên.
- Human-reference ban đầu mới hoàn tất 88/200; 20 case calibration có xung đột
  hoặc không chắc chắn vẫn chờ expert. Đây là dependency chưa đóng, không
  được trình bày như double-blind gold.
- Bộ old dataset và augmented dataset hiện được giữ cho benchmark/so sánh.
  Chưa có thủ tục nào hợp nhất chúng vào training set chính; chỉ được làm việc
  đó sau khi đối chiếu schema, nguồn gốc, leakage và phân phối.

### SẼ XỬ LÝ — kế hoạch chưa thực thi

1. Dùng hoặc điều chỉnh UI để human-review 1.978 record trong queue; mọi edit
   phải có annotator, timestamp, lý do và delta AI-before/human-after.
2. Expert adjudicate toàn bộ disagreement/uncertainty và khóa rõ ranh giới
   `Chất lượng`--`Hiệu năng & Trải nghiệm`, `Đúng mô tả`--`Tính xác thực`,
   sensory evidence và câu reward/“tặng shop 5 sao”.
3. Phát hành một version mới human-verified; không sửa âm thầm hay ghi đè
   release v1. Chạy lại schema/evidence validator, ledger joins, checksum và
   phân bố nhãn sau adjudication.
4. Tạo một benchmark blind bằng annotator B độc lập; tính agreement/IAA trước
   adjudication và báo riêng kết quả representative/challenge. AI-assisted
   confirmation không được dùng thay IAA.
5. Khóa group-aware train/dev/test split để product, duplicate family,
   template family và human-reference group không rò rỉ giữa các split.
   Quyết định minh bạch phần nào là gold, silver/pseudo hoặc chỉ unlabeled.
6. Đối chiếu old dataset và augmented dataset về aspect taxonomy, polarity,
   evidence, nguồn và duplicate/leakage; chỉ dùng làm benchmark hoặc ablation
   đúng vai trò đã khai báo, không trộn ngầm với dữ liệu mới.
7. Sau khi dữ liệu đạt gate, chạy baseline và mô hình chính, ablation,
   nhiều seed, confidence interval/significance test, error analysis và
   external benchmark; chưa có bước nào trong mục này được tính là đã chạy.
8. Neo manifest/checksum cuối vào signed Git tag hoặc kho immutable/DOI, rồi
   chuyển số liệu đã thực thi vào các phần Dataset, Annotation, Experiments và
   Limitations của paper.

### Quyết định, giới hạn và phụ thuộc kế tiếp

- **Quyết định hiện hành:** Giữ JSONL là canonical; giữ nguyên release v1;
  coi 5.000 record là AI pseudo-label chờ người xác minh; không gọi chúng là
  human gold và không trộn output từ model screening.
- **Kết quả tài liệu của task:** Protocol đã ghi rõ model thật sự tạo nhãn,
  cấu hình production, inventory đã/đang/sẽ xử lý và ranh giới giữa kết quả
  đo được với kế hoạch. Markdown nguồn được render lại thành DOCX và DOCX
  được kiểm tra khả năng chuyển ngược sang plain text bằng Pandoc.
- **Giới hạn:** Diagnostic chỉ có 20 clear case từ nguồn assisted; semantic
  audit cũng do AI thực hiện; 1.978 queue chưa human-check; human-reference
  chưa double-blind/đủ 200; chưa có IAA, expert-final label hay downstream
  experiment. Vì vậy chưa được suy diễn chất lượng Q1 chỉ từ technical
  validity hoặc diagnostic gate.
- **Phụ thuộc kế tiếp:** Human verification/adjudication là bước bắt buộc
  tiếp theo. Mọi task vật chất sau đây tiếp tục phải được thêm thành task-log
  mới, ghi rõ planned versus executed, rồi render và round-trip kiểm tra DOCX
  theo quy định repository.

## TASK-20260728-021 — Gán nhãn toàn bộ 8.976 review còn lại trong safe-frame

- **Trạng thái:** **ĐÃ THỰC THI VÀ PHÁT HÀNH.** Tranche continuation có đúng
  8.976/8.976 record, primary run đã `SEALED_REPLAY_VALID`, semantic audit đã
  đóng băng, final release đã được independent validator trả `VALID`. Không
  có job crawl/label nền nào còn chạy.
- **Mục tiêu:** Thực hiện yêu cầu bắt buộc của người dùng: gán nhãn phần còn
  lại bằng đúng production configuration đã dùng cho tranche 5.000; không
  ghi đè release cũ, không đưa human-reference/leakage group vào LLM, không
  đổi prompt/schema/model để tăng tốc và không sửa/xóa `data/raw/`.
- **Phân biệt phạm vi:** “Phần còn lại” được thực thi là toàn bộ **8.976
  record label-eligible** còn lại trong safe-frame 13.976, sau khi trừ 5.000
  record đã phát hành. 6.646/20.622 clean-core record thuộc 181
  human-reference/reserved leakage group vẫn được giữ ngoài labeling có chủ
  đích; chúng không phải record bị bỏ quên. Old dataset và augmented dataset
  không được đọc, trộn hay sửa trong task này.

### Đầu vào, selection và chống leakage — đã thực thi

- Nguồn vẫn là curation release 2.1.2
  `lazada-vi-absa-curation-e42d6c5319faedd3`, clean-core 20.622; cùng
  Guideline V2 2.0.0, human calibration DRAFT 88 complete-valid, 68 clear
  calibration case, 20 case loại chờ expert và cùng reservation ledger 6.646
  record.
- Script `prepare_ai_annotation_tranche.py` được mở rộng theo fail-closed
  continuation selection spec `absa-pseudolabel-selection/1.1.0`. Tùy chọn
  `--exclude-package` xác minh toàn bộ checksum closure, source/reference
  binding, ordered membership và final publication của tranche trước trước
  khi loại 5.000 record đó. Tranche mới là `tranche-0002`, ID
  `absa-ai-tranche-99d8e67fe7de83bb`.
- Kết quả join: safe-frame trước exclusion = 13.976; prior published =
  5.000; remainder = 8.976; exact human-reference overlap = 0; reserved-group
  overlap = 0; prior-tranche `sample_id` overlap = 0; prior-tranche
  review-text-hash overlap = 0. Hợp hai tranche có đúng 13.976 sample ID và
  13.976 text hash duy nhất. Ordered membership SHA-256 của tranche-0002 là
  `571d7340e20eac62c3b56818b86abd405f0fabdaf6b27247b0c5a6af00e88b7a`;
  prepare-manifest SHA-256 là
  `b0e2110eb345670c83b6a80969ed5b0a0fb8263d1ee7936a594fb5ffdf0fd0c1`.
- Package chuẩn bị nằm tại
  `data/annotations/absa_ai_remainder_8976_v1_20260728/`. Selection ledger
  phủ toàn safe-frame và phân biệt `SELECT_TRANCHE_0002`,
  `EXCLUDE_ALREADY_PUBLISHED_TRANCHE`; không xóa record nào khỏi nguồn.

### Cấu hình LLM bắt buộc và diagnostic — đã thực thi

- **Model tạo toàn bộ 8.976 nhãn mới:** `gpt-5.6-terra`, backend `codex`,
  endpoint logic `codex-cli://local-authenticated-session`, reasoning effort
  `medium`, batch size 20, 8 worker, tối đa 3 retry, timeout 420 giây,
  max-output setting 4.096 và không bật thinking override. Không có fallback
  model/provider.
- Prompt vẫn là `absa-ai-compact-v1.0.0`, system-prompt SHA-256
  `4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`;
  output schema vẫn có SHA-256
  `0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`.
  Final manifest của tranche 5.000 và tranche 8.976 xác nhận
  `execution_config` bằng nhau hoàn toàn trên backend, model, reasoning,
  prompt version/hash, schema hash và thinking mode.
- Diagnostic mới chạy lại trên đúng 20 holdout bằng cùng configuration và
  `PASS` toàn bộ gate: status exact 19/20 = 0,95; full-vector exact 12/20 =
  0,60; aspect-cell exact 171/180 = 0,95; mentioned precision 0,92, recall
  0,9388, F1 0,9293; polarity exact 44/46 = 0,9565. Diagnostic run-manifest
  SHA-256 là
  `f1cf4e02e9a9e65d6f2db2c0bfd31cf97ac8f842980824910393ee8729c24507`.
  Đây vẫn chỉ là alignment gate với AI-assisted human-confirmed holdout,
  không phải accuracy/IAA.

### Primary generation, resume và replay seal — đã thực thi

- Invocation chính ghi 8.970 record hợp lệ và để 6 record fail-closed sau
  validator retry. Resume pass thứ nhất tự bỏ qua 8.970 record đã hợp lệ, phục
  hồi 5/6; resume pass thứ hai phục hồi record cuối. Không có nhãn nào được
  sửa tay. Kết quả cuối: 8.976/8.976, missing ID = 0.
- Label summary của tranche mới: 8.229 `LABELED`, 616 `ESCALATE`, 131
  `REJECT_NON_REVIEW`; 22.441 mentioned aspect-cell; 952 mixed aspect-cell;
  1.691 review-level multi-polarity. Có 6 conservative case-only
  normalization repair được giữ trong ledger; không dùng semantic
  prefix/suffix trimming.
- Sealer replay lại stored response với frozen input, examples,
  prompt/schema/model config và record-attempt binding. Primary seal xác nhận
  672 attempt: 449 `VALID`, 223 `PARTIAL_VALID`; 6 failure marker đều
  `recovered`, unrecovered failure = 0; 8.976 record gắn với unique valid
  attempt. Primary run-manifest SHA-256 là
  `341871388ecdcdbfeed594dbe4d1c605813266a8ce7bd216e43cbe8c97dffe2f`.

### Sự cố provenance khi resume và cách xử lý — đã thực thi, không giấu

- Runner pre-fix tạo `batch_id` chỉ từ frozen target/lineage. Khi cùng một
  record tiếp tục lỗi qua hai resume invocation, attempt-01/02 mới có thể ghi
  đè file cùng tên, trong khi attempt-03 cũ còn lại; sealer đúng đắn đã từ
  chối chuỗi “attempt sau terminal”. Đây là lỗi bookkeeping provenance, không
  phải lỗi annotation.
- Runner được sửa để bind batch lineage với `invocation_id` gồm tranche,
  scope, timestamp và process ID; resume sau này không tái sử dụng attempt
  directory. Script `archive_ai_run_resume_collisions.py` phát hiện
  chronological inversion, sao lưu byte-for-byte stale attempt và Codex
  output sống sót vào recovery archive, xác minh hash rồi mới loại bản stale
  khỏi active replay tree. Không có label mutation.
- Recovery manifest SHA-256 là
  `4df5bfa5a966e7ba257b59e3bdd2a98c249c545e925063497948a3375827b1df`;
  checksum-ledger SHA-256
  `76fb95ac2127720aec324abaadc011cfd10de43c9622f72298db8576a885ff2c`.
  Recovery bridge khóa bản runner pre-fix với frozen package và runner fixed
  với runtime sealer trước khi cho seal.
- **Giới hạn provenance phải công bố:** attempt file đã bị ghi đè trước hotfix
  không thể tái dựng. Recovery archive chỉ bảo tồn stale attempt còn sống và
  failure marker bảo tồn last pre-resume error. Tuy nhiên mọi final record
  hiện tại đều replay được từ một unique valid stored attempt; sealer không
  miễn kiểm tra annotation/evidence vì sự cố này.

### Semantic audit và final release — đã thực thi

- Tạo audit code/schema mới, chọn deterministic 60 record không trùng nhau:
  10 reject, 10 escalate, 10 neutral, 10 mixed, 10 nhiều-aspect và 10 clear.
  Auditor là `gpt-5.6-terra`, reasoning `medium`, batch 20, ba worker, đọc
  Guideline V2 và annotation đã đóng; audit chỉ ghi severity/reason, không sửa
  nhãn.
- Kết quả audit: 51 `NO_MATERIAL_ISSUE`, 7 `MAJOR`, 2
  `MINOR_OR_BOUNDARY`; label mutation = 0. Major tập trung ở 5/10 escalate,
  1/10 neutral và 1/10 reject; clear-random và nhiều-aspect đều 10/10 không
  thấy lỗi vật chất trong sample này. Audit manifest SHA-256
  `ba1bf2b4145b0ae0c8fee0182fc62c5f2f931aa5794fe5afa860502c8385bf31`.
  Vì auditor là AI và sample cố ý oversample ca khó, các tỷ lệ này không phải
  corpus accuracy, human accuracy hay IAA.
- Final release nằm tại
  `data/annotations/absa_ai_remainder_8976_v1_20260728/final/`. Canonical là
  `ai_pseudo_labels.jsonl`; ngoài ra có CSV projection, 8.976-row decision
  ledger, human-review queue 3.592 row, sealed diagnostic/primary provenance,
  audit artifact và checksum closure. Trạng thái là
  `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`, terminal state
  `PUBLISHED_PENDING_HUMAN_VERIFICATION:8976`.
- Independent validator chạy lại từ release đã xuất và trả `VALID`, record =
  8.976, queue = 3.592, exact-reference overlap = 0, reserved-group overlap =
  0. Final manifest SHA-256
  `c3c5f6e77553cd148258b5808ce756881ed6e03c1351277be1d92329666fbe47`;
  final `SHA256SUMS.txt` SHA-256
  `957c56e6d992d1f06bbb211d107ac55710ceeee6f0068c48ff94917dcbdf527e`.

### Tổng hợp sau hai tranche — kết quả hiện tại

| Chỉ số | Tranche 5.000 | Tranche 8.976 | Safe-frame union |
|---|---:|---:|---:|
| Record | 5.000 | 8.976 | 13.976 |
| `LABELED` | 4.598 | 8.229 | 12.827 |
| `ESCALATE` | 320 | 616 | 936 |
| `REJECT_NON_REVIEW` | 82 | 131 | 213 |
| Mentioned aspect-cell | 12.402 | 22.441 | 34.843 |
| Mixed aspect-cell | 556 | 952 | 1.508 |
| Review-level multi-polarity | 1.004 | 1.691 | 2.695 |
| Human-review queue | 1.978 | 3.592 | 5.570 |

- Hai final JSONL có 13.976 sample ID và 13.976 review-text hash duy nhất;
  overlap giữa hai release = 0. Hai execution config bằng nhau hoàn toàn.
  Queue union cũng không trùng nguồn vì membership hai tranche không giao
  nhau.
- **Trạng thái hiện tại:** Toàn bộ label-eligible safe-frame đã có AI
  pseudo-label. 6.646 reserved-group record vẫn chưa được LLM label theo chủ
  đích chống leakage. Không có bước human verification mới nào được thực
  hiện trong task này.

### Code, kiểm thử, quyết định và bước kế tiếp

- **Code/config đã thay đổi:** continuation selection và prior-release
  verification; dynamic tranche decision/expected-record CLI; unique resume
  invocation lineage; collision recovery/archive bridge; generic 60-record
  semantic-audit runner và strict audit schema; README cập nhật hai release.
- **Kiểm thử đã chạy:** 194/194 Python test và 8/8 JavaScript test đạt;
  `py_compile` đạt cho preparation, runner, recovery, sealer, audit,
  finalizer và validator. Prepared-package validator xác nhận union/overlap;
  final release validator được chạy độc lập lần hai.
- **Quyết định đã thực thi:** Giữ hai release versioned bất biến thay vì ghi
  đè hoặc trộn âm thầm; JSONL là canonical; 8.976 label mới vẫn là pseudo;
  không label 6.646 reserved group; không gọi diagnostic/audit là human
  accuracy; không dùng old/augmented dataset trong generation.
- **Giới hạn khoa học:** Cả hai tranche dùng cùng một LLM và calibration
  AI-assisted nên có correlated error/confirmation bias. Audit 60 vẫn do AI.
  Human-reference mới 88/200, chưa double-blind; chưa có annotator B, IAA,
  expert adjudication hay human-verified final label. Phân bố tự nhiên vẫn
  chịu ảnh hưởng lớn từ review 5 sao.
- **KẾ HOẠCH CHƯA THỰC THI:** (1) hợp nhất UI assignment cho 5.570 queue
  record nhưng vẫn giữ child-tranche provenance; (2) human-check ưu tiên mọi
  audit major/minor, E/R, uncertainty, neutral, mixed và marketing-like; (3)
  expert adjudicate disagreement và xuất release version mới, không sửa hai
  release pseudo hiện tại; (4) hoàn tất 200 human-reference và annotator B
  blind, tính IAA trước adjudication; (5) sau đó mới khóa gold/silver policy,
  group-aware train/dev/test, benchmark old/augmented dataset, huấn luyện và
  đánh giá mô hình; (6) neo final digest vào signed/immutable external record.
- **Cập nhật tài liệu của task:** Mục này ghi objective, input, code/method,
  kết quả đo, sự cố, quyết định, giới hạn và next dependency. Markdown được
  render lại thành DOCX và DOCX được Pandoc round-trip sau khi task hoàn tất.

## TASK-20260728-022 — Đối soát 31.928 record nguồn với 5.000 + 8.976 pseudo-label

- **Trạng thái:** Đã thực thi đối soát chỉ đọc và cập nhật tài liệu; không thay
  đổi nhãn, không di chuyển/xóa record và không sửa `data/raw/`.
- **Mục tiêu:** Giải thích chính xác vì sao tập canonical ban đầu có 31.928
  record nhưng hai tranche pseudo-label chỉ có 5.000 và 8.976 record; phân biệt
  corpus nguồn, clean-core và label-safe frame để tránh diễn giải sai phạm vi
  của hai release.
- **Đầu vào đã đọc:** Manifest của curation release
  `data/releases/lazada_vi_absa_curation_v2_1_2_20260725/manifest.json`;
  reservation ledger
  `data/annotations/human_reference_v1_20260726/private/group_reservations.jsonl`;
  prepare/final manifest của hai package
  `absa_ai_tranche_5000_v1_20260727` và
  `absa_ai_remainder_8976_v1_20260728`.

### Phép đối soát — đã thực thi

Phép tính phải được đọc theo cấu trúc lồng nhau, vì 6.646 record bảo lưu là
**tập con của 20.622 clean-core**, không phải một partition bổ sung ngoài
31.928:

```text
31.928 canonical parent
├── 20.622 clean-core = KEEP + KEEP_CLEANED
│   ├── 6.646 reserved leakage-group records
│   │   ├── 200 human-reference rows
│   │   └── 6.446 non-reference rows cùng các leakage group
│   └── 13.976 label-safe records
│       ├── 5.000 pseudo-label ở tranche-0001
│       └── 8.976 pseudo-label ở tranche-0002
├── 11.166 QUARANTINE
└── 140 EXCLUDE_AUTO duplicate aliases
```

Các đẳng thức kiểm tra closure:

```text
31.928 = 20.622 + 11.166 + 140
20.622 = 6.646 + 13.976
13.976 = 5.000 + 8.976
31.928 = 13.976 + 6.646 + 11.166 + 140
```

| Lớp dữ liệu | Số record | Diễn giải thực tế |
|---|---:|---|
| Canonical parent | 31.928 | Toàn bộ record nguồn đã vào curation release |
| Clean-core | 20.622 | `KEEP` 20.429 + `KEEP_CLEANED` 193 |
| Reserved trong clean-core | 6.646 | 200 reference + 6.446 hàng xóm trong 181 leakage group |
| Label-safe frame | 13.976 | Toàn bộ phạm vi được phép pseudo-label ở chính sách hiện tại |
| Tranche-0001 | 5.000 | Đã pseudo-label và phát hành chờ human verification |
| Tranche-0002 | 8.976 | Đã pseudo-label và phát hành chờ human verification |
| `QUARANTINE` | 11.166 | Chưa kết luận là rác; cần human curation trước khi gán nhãn ABSA |
| `EXCLUDE_AUTO` | 140 | Duplicate aliases đã xác nhận, không được tính như mẫu độc lập |

- **Kết quả đo:** Reservation ledger có đúng 6.646 dòng, 6.646 `sample_id`,
  181 `leakage_group_id`; trong đó 200 dòng có `is_reference_row=true` và
  6.446 dòng có `is_reference_row=false`. Hai tranche phủ đúng 13.976
  label-safe record và không giao nhau theo `sample_id` hoặc review-text hash.
- **Giải thích phạm vi đã thực thi:** Cụm từ “phần còn lại” ở TASK-021 được
  triển khai theo nghĩa **phần còn lại của label-safe frame**, tức
  `13.976 - 5.000 = 8.976`, chứ không phải phần còn lại của toàn bộ 31.928
  canonical parent. Đây là lựa chọn bảo thủ để không đưa record cách ly hoặc
  record thuộc nhóm reference/evaluation vào pseudo-label một cách âm thầm.
- **Không mất dữ liệu:** 11.166 record `QUARANTINE` và 6.646 record reserved
  vẫn còn nguyên trong các release/ledger tương ứng. 140 duplicate aliases
  được lưu bằng decision ledger nhưng không được coi là 140 quan sát độc lập.

### Quyết định, giới hạn và phụ thuộc kế tiếp

- **Quyết định đã thực thi:** Không tự động gán nhãn 11.166 record
  `QUARANTINE`; không phá reservation của 181 leakage group; không đưa 140
  duplicate alias trở lại tập mẫu; không thay đổi hai release pseudo-label đã
  đóng gói.
- **Giới hạn:** `QUARANTINE` không đồng nghĩa toàn bộ 11.166 record đều vô
  dụng. Một phần có thể là review hợp lệ nhưng bị rule bảo thủ giữ lại; chưa có
  human audit nên chưa thể báo số recoverable. Tương tự, 6.446 non-reference
  neighbor không nhất thiết là review xấu; chúng được bảo lưu vì quan hệ nhóm
  với human-reference, không phải vì chất lượng văn bản.
- **KẾ HOẠCH CHƯA THỰC THI:** (1) audit phân tầng 11.166 record
  `QUARANTINE`, human-confirm và phát hành một curation release mới cho phần
  recoverable; (2) chỉ sau khi khóa chính sách benchmark mới quyết định có giữ
  toàn bộ 6.646 group reservation hay giải phóng một phần 6.446
  non-reference rows bằng split/group-aware policy; (3) 140 duplicate aliases
  tiếp tục chỉ tồn tại trong provenance/decision ledger, không dùng làm mẫu
  huấn luyện độc lập.
- **Next dependency:** Muốn mở rộng vượt 13.976 pseudo-label mà vẫn đúng quy
  trình khoa học, bước kế tiếp là human curation đối với `QUARANTINE` và khóa
  chính sách group-aware benchmark; không phải chạy LLM trực tiếp lên toàn bộ
  phần còn lại.

## TASK-20260728-023 — Tái gán nhãn label-blind cho old dataset

- **Trạng thái:** Đã hoàn tất source release label-blind, generation
  9.772/9.772, semantic audit, replay seal, canonical release, projection
  10.105 dòng, independent validation, kiểm thử và cập nhật tài liệu. Không
  sửa/xóa `data/raw/` hoặc workbook old lịch sử.
- **Mục tiêu:** Tái gán nhãn toàn bộ old dataset bằng đúng model/prompt/schema
  configuration đã dùng cho tranche 5.000 và 8.976, không cho bất kỳ nhãn cũ
  nào đi vào target, selection, prompt calibration hay validation. Xuất release
  versioned mới và projection trở lại layout XLSX cũ mà không ghi đè nguồn
  lịch sử.
- **Đầu vào:** 10 workbook
  `legacy/data/old_dataset/test_flow_reviews_part*_labeled.xlsx`; Guideline V2
  2.0.0; cùng 88 human-confirmed calibration record, cùng diagnostic split,
  cùng compact prompt và output schema của release 8.976.

### Inventory, loại nhãn cũ và deduplication — đã thực thi

- Audit read-only xác nhận 10.105 dòng có `reviewContent` không rỗng, không
  phải 11.000 chính xác. Có 9.773 exact-unique text và 9.772 unique key sau
  NFKC + casefold + collapse whitespace + strip.
- Tổng cộng 90.945 ô thuộc chín cột nhãn lịch sử; 88.137 ô không rỗng. Toàn bộ
  **giá trị** của các ô này bị loại khỏi clean-core và LLM target. Chỉ SHA-256
  một chiều của vector nhãn được giữ trong source-row decision ledger để chứng
  minh quyết định stripping mà không truyền đáp án cũ sang annotator.
- 10.105 dòng được đóng bằng ledger: 9.772
  `CANONICAL_REPRESENTATIVE_LABELS_STRIPPED` và 333
  `DUPLICATE_ALIAS_MAP_TO_CANONICAL`. LLM chỉ nhận 9.772 text duy nhất; khi
  phát hành, 333 alias sẽ nhận annotation của canonical review thay vì tốn
  lời gọi lặp hoặc tạo leakage.
- Các workbook lịch sử **không bị sửa/xóa**. Cụm “xóa nhãn cũ” được thực thi
  theo nghĩa label-blind re-annotation: nhãn cũ không tồn tại trong dataset đầu
  vào mới và không được dùng để dạy/chọn/chấm LLM. Giữ workbook gốc bất biến là
  yêu cầu chain-of-custody; các workbook đầu ra sẽ là file mới.
- Source release đã đóng băng tại
  `data/releases/legacy_old_reviews_label_blind_v1_20260728/`, release ID
  `legacy-old-label-blind-e28aa62584100f31`. Manifest SHA-256:
  `635a3ba6c0476eeec1d6a326b4ff4aab5f26f31092d8b1ba027268b1df9535a3`;
  checksum ledger SHA-256:
  `a222ebbd92cfc2feb85f27f554b7579613031b1194a777ef3b2ca72718b1b815`.
  Independent closure check xác nhận 10.105 source row, 9.772 clean-core
  target, 333 alias và zero historical label field/value trong target.

### Package, cấu hình và diagnostic — đã thực thi

- Work package:
  `data/annotations/absa_legacy_old_relabel_9772_v1_20260728/`, tranche ID
  `absa-ai-tranche-11c861d402984cd4`, target 9.772. Prepared manifest
  SHA-256:
  `473fd040c325e53ce7ac4db0a5d5a8d22c8352bb3c98934744973acf9ffd2220`;
  input checksum-ledger SHA-256:
  `12f684ca554ce7759635311757c481334fd816820ccee2bbdc81a038d6323c0d`.
- Existing independent prepared-package validator trả
  `VALID_PREPARED`: 9.772 rank liên tục, 9.772 sample ID/text hash duy nhất,
  blind/private/selection/source joins đóng kín và không có sample-ID overlap
  với human-reference reservation.
- **Cấu hình khóa và đã dùng:** backend `codex`, model `gpt-5.6-terra`,
  reasoning `medium`, batch size 20, 8 worker, tối đa 3 retry, timeout 420
  giây, max-output setting 4.096, thinking override `null`; prompt version
  `absa-ai-compact-v1.0.0`, system-prompt SHA-256
  `4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`,
  schema SHA-256
  `0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`.
  Machine comparison xác nhận execution-config diagnostic mới bằng chính xác
  primary execution-config của release 8.976.
- Diagnostic chạy lại trên đúng frozen 20-record holdout và `PASS` mọi gate:
  status exact 19/20 = 0,95; full-vector exact 12/20 = 0,60; aspect-cell exact
  171/180 = 0,95; mentioned precision 0,9375, recall 0,9184, F1 0,9278;
  polarity exact khi hai phía cùng mentioned 43/45 = 0,9556. Đây là alignment
  với AI-assisted human-confirmed holdout, không phải human accuracy hay IAA.

### Cross-corpus overlap audit — đã thực thi

- Machine-readable audit:
  `docs/audits/legacy_old_cross_corpus_overlap_20260728.json`, SHA-256
  `e347646358f41c91b016b00ff6a576d126abfe6a1b60cbce746c299dda033a2b`.
- Old target có zero exact/normalized overlap với 88 prompt-calibration
  record, vì vậy không có target-answer leakage qua in-context example.
- Có 2 exact text-hash overlap với 6.646 reserved leakage-group record của
  human-reference corpus; có 1 exact và 5 normalized overlap với union hai
  pseudo-label release 5.000 + 8.976. Không nhãn nào được import từ các
  overlap này, nhưng chúng không được đếm như independent benchmark evidence
  và phải nằm cùng group hoặc bị loại khi split/evaluate.
- **Quyết định khoa học:** Vì old dataset đang được chính LLM tái gán nhãn, bản
  mới là silver/pseudo-label dataset chờ human verification, không còn là một
  human-gold benchmark độc lập chỉ nhờ tên “old dataset”.

### Primary, publication và kiểm thử

- **Primary đã thực thi:** Invocation chính ghi 9.770/9.772 record hợp lệ; hai
  record fail-closed vì model gắn evidence cho aspect `Đúng mô tả` có label
  absent. Resume bằng đúng config tự bỏ qua 9.770 record, dùng 2 provider call
  và phục hồi cả hai; missing ID cuối = 0. Không có label sửa tay.
- Replay seal trả `SEALED_REPLAY_VALID`: 679 attempt gồm 490 `VALID`, 185
  `PARTIAL_VALID`, 4 `SCHEMA_INVALID`; 2 failure marker `recovered`, 0
  unrecovered failure, 9.772 record gắn với unique valid attempt, normalization
  repair = 0. Diagnostic run-manifest SHA-256:
  `418ed67a97e9bcc1e5500f99da4fce3b532b8fc83d649ad1bc92e37ba3c32e43`;
  primary run-manifest SHA-256:
  `07f2c77ed77820a703bfa1870e68b76c5a4b4c8b4a6ff2e9bf0a35b9bcd424dc`.
- Canonical label summary trên 9.772 unique review: 8.553 `LABELED`, 950
  `ESCALATE`, 269 `REJECT_NON_REVIEW`; 18.942 mentioned aspect-cell, 667
  mixed aspect-cell và 1.551 review-level multi-polarity. Đây là phân bố tự
  nhiên sau labeling, không fit về phân bố nhãn old.

### Semantic audit và canonical release — đã thực thi

- Audit tool ban đầu dừng fail-closed vì còn hard-code target 8.976. Code được
  sửa để đọc positive integer `target_records` từ prepared manifest rồi kiểm
  tra sealed primary inventory theo giá trị đó; không thay đổi sampling,
  model/prompt/schema hoặc nhãn. `py_compile` và semantic-audit unit test đạt
  trước khi chạy lại.
- Semantic audit deterministic 60 record, 10 record cho mỗi stratum reject,
  escalate, neutral, mixed, high-aspect và clear, trả 48
  `NO_MATERIAL_ISSUE`, 3 `MINOR_OR_BOUNDARY`, 9 `MAJOR`; label mutation = 0.
  Major theo stratum: neutral 3, escalate 2, mixed 2, high 1, clear 1, reject
  0. Vì sample cố ý oversample ca khó và auditor vẫn là AI, các tỷ lệ này
  không phải corpus accuracy, human accuracy hay IAA. Audit manifest SHA-256:
  `023e798c0eb894613c5b6729eadba0447c223d77906b112809509744d42c8e6d`.
- Canonical release:
  `data/annotations/absa_legacy_old_relabel_9772_v1_20260728/final/`, trạng
  thái `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`. Independent standard
  validator trả `VALID`, 9.772 record, terminal state
  `PUBLISHED_PENDING_HUMAN_VERIFICATION:9772`, reference sample-ID overlap = 0,
  reserved sample-ID overlap = 0.
- Human-review queue có 3.570 unique review. Queue bắt buộc gồm
  escalate/reject, neutral, mixed, uncertainty, marketing-like, evidence-risk
  và các risk flag khác; thêm deterministic/stratified audit coverage.
  Human verification hoàn tất = 0.
- Final manifest SHA-256:
  `350be6f16cebdb588c779cc4c1befdc8c42db0c29fd69f55394cd162894a94df`;
  final checksum-ledger SHA-256:
  `4bf0b4f75e6158a0f6e07d9aacbd78e48140c5c186cbd4274c54c8d82d7a75c5`.

### Projection đủ 10.105 dòng và independent legacy validator — đã thực thi

- Projection:
  `data/annotations/absa_legacy_old_relabel_9772_v1_20260728/legacy_projection/`.
  `legacy_old_relabel_10105.jsonl` có đúng 10.105 source row; 10 workbook mới
  dưới `xlsx_10col_new_labels/` có tổng đúng 10.105 dòng và đúng 10 header cũ.
  Không ghi đè 10 workbook lịch sử.
- 333 duplicate alias nhận annotation của 9.772 canonical review qua
  `canonical_sample_id`; vì vậy **10.105 không phải số review được LLM gán nhãn
  độc lập**. Source-row status sau expansion là 8.831 `LABELED`, 980
  `ESCALATE`, 294 `REJECT_NON_REVIEW`; khác canonical counts vì alias lặp.
- Projection manifest xác nhận `historical_label_values_reused=false` và
  `source_workbooks_modified=false`. Projection manifest SHA-256:
  `4483cd5e7e41da756f6f824ebf597d215a2d85b5e4be24f804828f31f4a37dc5`;
  checksum-ledger SHA-256:
  `04e99daf070912f56fc9c8c05928a0d4ec4d66c96636d2941acbb114d8d2d882`.
- Independent legacy validator replay standard release validator, source
  checksum closure, 10 workbook nguồn, label scrub, blind target uniqueness,
  exact config comparison, 10.105 alias joins và từng label trong 10 workbook
  mới. Kết quả `VALID`: canonical 9.772, projection JSONL 10.105, XLSX rows
  10.105, old-label value trong LLM target = 0, source modified = false,
  diagnostic/primary config exact-match prior tranche = true. Report:
  `docs/audits/legacy_old_reannotation_validation_20260728.json`, SHA-256
  `e8e251c41281c0a02d7ca95067adfccecd4bfc8ff675b6f4d9e85293bbe6bcf8`.
- **Kiểm thử cuối task:** 194/194 Python unit test và 8/8 JavaScript test đạt;
  preparation, publisher, legacy validator, overlap audit và generic semantic
  audit script đều qua `py_compile`.

### Quyết định, giới hạn và next dependency

- **Quyết định đã thực thi:** Giữ workbook old gốc bất biến làm provenance;
  nhãn old không đi vào bất kỳ target/prompt/selection/validation nào; chỉ
  label 9.772 text duy nhất; không fit phân bố; không dùng augmented dataset;
  canonical JSONL là artifact nghiên cứu, XLSX là projection tương thích;
  mọi nhãn mới vẫn là pseudo-label chờ human verification.
- **Giới hạn khoa học:** Calibration là AI-assisted human-confirmed và
  generation/audit dùng cùng họ LLM, nên còn confirmation bias/correlated
  error. 9 `MAJOR` và 3 boundary issue trong difficult-strata audit cần human
  adjudication. 3.570 queue chưa được human-check. Hai old review exact-overlap
  reserved reference và năm review normalized-overlap current pseudo union
  không được tính như independent benchmark evidence. Old dataset mới không
  trở thành human gold chỉ vì đã thay nhãn cũ.
- **KẾ HOẠCH CHƯA THỰC THI:** (1) human-check trước 12 audit issue, sau đó toàn
  bộ 3.570 queue; (2) expert adjudicate disagreement và phát hành release mới,
  không sửa in-place pseudo release; (3) nếu dùng old làm benchmark, cần
  annotator độc lập/double-blind, IAA và group-aware exclusion của mọi
  cross-corpus overlap; (4) audit augmented dataset riêng, giữ synthetic chỉ
  ở train sau khi split real data; (5) chỉ khóa train/dev/test sau khi gold và
  silver policy được chốt.
- **Next dependency:** Human verification/adjudication của queue và quyết định
  vai trò old dataset (`silver training data` hay `human-reannotated
  benchmark`). Không dùng ngay 9.772 pseudo-label làm ground-truth Q1.

## TASK-20260728-024 — Đối soát nguyên nhân 11.166 record bị quarantine

- **Trạng thái:** Đã thực thi audit chỉ đọc trên frozen curation release; không
  thay đổi status, text, nhãn hoặc `data/raw/`.
- **Mục tiêu:** Giải thích bằng decision ledger vì sao 11.166/31.928 canonical
  record không đi vào clean-core và phân biệt `QUARANTINE` với
  `EXCLUDE_AUTO`.
- **Đầu vào:** `manifest.json` và `quarantine.jsonl` của
  `data/releases/lazada_vi_absa_curation_v2_1_2_20260725/`, rule version
  2.1.2. Toàn bộ 11.166 JSONL record được đếm lại theo
  `curation.primary_reason` và `curation.reason_codes`.

### Kết quả — đã thực thi

| Primary reason loại trừ lẫn nhau | Record | Tỷ lệ trong quarantine |
|---|---:|---:|
| `PLATFORM_TEMPLATE` | 9.874 | 88,43% |
| `REWARD_DISCLOSURE` | 661 | 5,92% |
| `POST_CLEAN_QUALITY` | 203 | 1,82% |
| `INTERNAL_REPETITION` | 159 | 1,42% |
| `NONREVIEW_SUSPECT` | 151 | 1,35% |
| `HARD_NONREVIEW` | 59 | 0,53% |
| `NEAR_DUPLICATE` | 34 | 0,30% |
| `PRIVACY` | 24 | 0,21% |
| `DUPLICATE` chưa giải quyết một-một | 1 | 0,01% |
| **Tổng** | **11.166** | **100%** |

- Nguyên nhân áp đảo là `PLATFORM_TEMPLATE`. Có 9.540 record mang flag
  `STRUCTURAL_CATALOGUE_NO_EXPERIENCE`; các mẫu điển hình là chuỗi câu
  liệt kê/preset chung như “bền”, “dễ vệ sinh”, “rất được khuyến nghị” nhưng
  không có chủ thể, sự kiện mua/dùng hay trải nghiệm cụ thể. Các flag phụ như
  `TEMPLATE_GLOBAL_CANDIDATE` 1.848, `TEMPLATE_PRODUCT_HIGH` 1.319,
  `TEMPLATE_GLOBAL_HIGH` 373 và `INTERNAL_CLAUSE_REPEAT` 1.713 có thể cùng
  xuất hiện trên một record nên **không được cộng** như partition.
- `REWARD_DISCLOSURE` gồm review xin xu/coin, vote/like hoặc khai báo nội dung
  được thêm chỉ để nhận thưởng; một số vẫn chứa đánh giá sản phẩm nên được giữ
  chờ người quyết định thay vì xóa toàn câu.
- `POST_CLEAN_QUALITY`, `INTERNAL_REPETITION`, `NONREVIEW_SUSPECT` và
  `HARD_NONREVIEW` gồm keyboard-smash/gibberish, câu lặp, trích dẫn phim/tin,
  Gboard clipboard, SMS nhà mạng và nội dung không phải trải nghiệm mua hàng.
- `PRIVACY` chứa phone/contact/URL đã redact hoặc quảng bá liên hệ;
  `NEAR_DUPLICATE` chỉ là ứng viên similarity chưa đủ bằng chứng tự động loại;
  một `DUPLICATE` còn mơ hồ giữa transport.
- 9.572 quarantine record đến từ `requests_cookie`, 1.592 từ
  `selenium_dom`, 2 từ `requests`. Vì cả transport đều có record bị giữ,
  quarantine không đơn thuần là lỗi riêng của Selenium.

### Diễn giải, quyết định và giới hạn

- **Tại sao không xóa:** Frozen policy chỉ cho `EXCLUDE_AUTO` khi duplicate đã
  được xác nhận một-một; hard-nonreview/privacy/post-clean-quality/template/
  near-duplicate dựa trên heuristic đều bị cấm auto-delete. Vì vậy 11.166
  record có `decision_source=MANUAL_PENDING`, `annotation_eligible=false`
  và được giữ nguyên để human curation.
- **Không đồng nghĩa 11.166 record đều là rác:** Rule
  `STRUCTURAL_CATALOGUE_NO_EXPERIENCE` rất bảo thủ và chi phối 88,4% nhóm này.
  Nó có thể giữ lại cả câu preset/catalogue thật lẫn review ngắn nhưng vẫn có
  polarity hữu ích. Do đó đây là vùng có nguy cơ false positive lớn nhất và
  không được báo như 11.166 non-review đã xác nhận.
- **Quyết định đã thực thi:** Không đưa thẳng quarantine vào LLM labeling,
  không đổi threshold/rule hậu nghiệm để làm tăng corpus, và không trộn các
  reason-code count chồng lấn với primary-reason partition.
- **KẾ HOẠCH CHƯA THỰC THI:** Human audit phân tầng, ưu tiên
  `PLATFORM_TEMPLATE`; tách `RECOVER_SUBSTANTIVE_REVIEW`,
  `REJECT_PURE_TEMPLATE/NONREVIEW` và `CLEAN_THEN_RECOVER` cho reward/privacy.
  Sau adjudication mới phát hành curation release mới và pseudo-label phần
  recoverable.
- **Next dependency:** Khóa protocol human audit và số mẫu theo từng primary
  reason. Chưa có recovery rate do chưa thực thi human curation, nên chưa thể
  nói còn bao nhiêu trong 11.166 sẽ được cứu.

## TASK-20260728-025 — Đối soát hành vi khi chạy lại crawler

- **Trạng thái:** Đã thực thi kiểm tra chỉ đọc code/config và hai lệnh
  `--dry-run`; chưa liên hệ Lazada, chưa mở Selenium, chưa tạo crawl run và
  không thay đổi `data/raw/`.
- **Mục tiêu:** Xác định khi chạy lại crawler thì target được hiểu thế nào,
  dữ liệu cũ có bị ghi đè không, quy trình lọc/deduplicate có đổi không và
  artifact mới được lưu ở đâu.
- **Đầu vào đã kiểm tra:** `configs/collector.toml`,
  `run_crawl_automatic.ps1`, `src/crawl.py`,
  `src/lazada_collector/{cli,history,storage}.py`, README và toàn bộ crawl
  history hiện có dưới `data/raw/`.
- **Phương pháp đã thực thi:** Đọc code tạo `crawl_id`/run directory, code
  nạp collection history và cross-run dedup; chạy offline:
  `lazada-collect crawl-scale --target-reviews 30000 --dry-run` và
  `lazada-collect crawl-dom-scale --target-reviews 30000 --dry-run`.

### Kết quả đo được và hành vi hiện tại

- Cả API và DOM dry-run đếm đúng **31.928** accepted record theo policy hiện
  hành `substantive_vi_v2`; API history báo thêm 21 record cũ/không hợp lệ
  không được tính vào target. Cookie loader xác nhận 32 cookie Lazada còn
  active tại thời điểm audit; giá trị cookie không được ghi vào dataset.
- `--target-reviews` là **tổng unique accepted review dưới output root**, không
  phải số review muốn cào thêm. Vì 31.928 > 30.000, chạy lại với target
  30.000 sẽ trả `already_satisfied`, `run_created=false` và không cào thêm.
  Muốn thêm đúng 10.000 từ mốc hiện tại thì target vận hành tương ứng là
  41.928; số thực nhận vẫn phụ thuộc platform/rate limit và quality filter.
- Mỗi invocation thật chưa đủ target tạo một `crawl_id` mới theo UTC và ghi
  vào `data/raw/YYYY-MM-DD/<crawl-id>/`, gồm `products.jsonl`,
  `reviews.jsonl`, `rejections.jsonl`, `manifest.json`. `exist_ok=false` và
  UUID trong `crawl_id` ngăn ghi đè run cũ. Supervisor hybrid có thể tạo nhiều
  run directory trong một lần gọi vì mỗi API/DOM cycle là một invocation.
- Resume là resume **logic qua history**, không mở lại và append vào run
  directory cũ: crawler quét mọi `reviews.jsonl`/`products.jsonl`/
  `rejections.jsonl` dưới cùng output root, tái dùng product cache và
  checkpoint DOM, đồng thời loại cross-run duplicate theo `review_id` hoặc
  normalized text key. API mặc định tránh product đã thử; DOM mặc định tránh
  product đã hoàn tất và có thể tiếp tục từ page checkpoint.
- Review mới vẫn qua cùng quality policy: tối thiểu 80 ký tự, 15 từ, unique
  word ratio 0,4, 8 meaningful word, quality score 0,55, tín hiệu tiếng Việt,
  foreign-script/encoding checks; rating filter 0 giữ phân bố tự nhiên.
  Rejection chỉ lưu review ID, text hash và quality measurements, không lưu
  rejected text.
- `run_crawl_automatic.cmd` mặc định dùng hybrid cookie API + Selenium DOM,
  ghi thêm supervisor log tại `logs/crawl-supervisor.log`, chuyển transport
  theo status và cooldown. Nó không bypass CAPTCHA/challenge.

### Quyết định, giới hạn và next dependency

- **Quyết định đã thực thi:** Không khởi động live crawl trong task giải thích
  này. Giữ nguyên raw append-only và các frozen release/annotation artifact.
- **Phạm vi deduplicate:** Cross-run dedup của crawler chỉ đọc output root
  `data/raw/`; nó không dùng old dataset, pseudo-label release hoặc human
  reference làm collection history. Cross-corpus overlap vẫn phải kiểm lại ở
  bước xây release.
- **Không tự động cập nhật downstream:** Crawl mới không sửa
  `data/releases/lazada_vi_reviews_v1_20260725/`, curation v2.1.2, quarantine
  hay bất kỳ pseudo-label package nào. Sau collection phải audit/freeze thành
  snapshot và versioned release mới, sau đó mới chạy cleaning, curation và
  annotation; không append trực tiếp vào frozen release cũ.
- **KẾ HOẠCH CHƯA THỰC THI:** Nếu mở collection round mới, trước tiên chạy
  supervisor `-ValidateOnly` với target tổng đã tính; sau đó live crawl; đóng
  tất cả manifest, audit provenance/duplicate/non-review, và phát hành corpus
  release ID mới.
- **Next dependency:** Người dùng quyết định số review muốn **cào thêm** và
  sampling frame của round mới. Với mốc đã đo 31.928, thêm 10.000 tương ứng
  target tổng 41.928.

## TASK-20260728-026 — Audit coverage từ raw tới curation và labeling

- **Trạng thái:** Đã thực thi audit chỉ đọc code, manifest và replay
  validator; không crawl, không thay đổi raw/release/nhãn.
- **Mục tiêu:** Trả lời liệu cleaning, quarantine và labeling đã được ghi lại
  đầy đủ chưa, và liệu review cào mới có tự động đi 100% qua toàn bộ luồng hay
  không.
- **Đầu vào:** Protocol hiện hành; manifest của parent release 31.928,
  curation v2.1.2 và hai pseudo-label release 5.000 + 8.976; CLI của
  `build_corpus_release.py`, `build_clean_release_v2.py`,
  `prepare_ai_annotation_tranche.py`; các validator release.

### Coverage snapshot 2026-07-25 — đã xác minh

- Raw inventory có 31.949 review record vật lý. Parent builder loại có ledger
  21 record policy `substantive_v1` cũ và tạo đúng 31.928 canonical record
  `substantive_vi_v2`; malformed = 0.
- Replay `validate_clean_release_v2.py` trả `VALID`: toàn bộ 31.928 parent
  record xuất hiện đúng một lần trong curation ledger và partition kín:

| Curation status | Record | Hành động |
|---|---:|---|
| `KEEP` | 20.429 | clean-core |
| `KEEP_CLEANED` | 193 | clean-core sau transformation |
| `QUARANTINE` | 11.166 | chờ human curation, không tự label |
| `EXCLUDE_AUTO` | 140 | confirmed duplicate |
| **Tổng** | **31.928** | **100% parent coverage** |

- Clean-core tổng cộng 20.622. Trong đó 6.646 record thuộc leakage group của
  human-reference được reserve; 13.976 safe-frame còn lại tạo thành hai
  pseudo-label release không giao nhau: 5.000 + 8.976 = 13.976. Do đó
  labeling không và không nên bằng 31.928.
- Trên 13.976 record đã đi qua LLM workflow, terminal status là 12.827
  `LABELED`, 936 `ESCALATE`, 213 `REJECT_NON_REVIEW`; tất cả vẫn là
  `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`, không phải human gold.
- Replay current validator cho tranche 8.976 trả `VALID`. Current validator
  từ chối tranche 5.000 tại
  `diagnostic sealing implementation binding mismatch` vì seal script hiện
  tại có SHA-256 khác bản đã dùng để đóng release. Đây là fail-closed
  implementation binding, không phải bằng chứng record/checksum hỏng. Replay
  bằng validator và seal script đã archive trong
  `final/provenance/software/` của chính tranche 5.000 trả `VALID`, đúng 5.000
  record, manifest SHA-256
  `bc01e6727b73e5962aa14affea4f3628917554572087a84da951ad69e03f5da1`.

### Trả lời về automation và bảo đảm 100%

- Cleaning, curation/quarantine, transformation ledger, duplicate ledger,
  annotation selection, generation config, review queue, checksum và
  limitation **đã được ghi trong protocol/DOCX và manifest từng release**.
- Tuy nhiên, chạy `run_crawl_automatic.cmd` chỉ kết thúc ở `data/raw/`.
  Repository hiện **chưa có một orchestrator end-to-end** tự phát hiện crawl
  hoàn tất rồi tự build parent, curation, prepare tranche, gọi LLM, seal và
  validate. Các bước downstream là những lệnh riêng và default config hiện
  còn gắn với release 2026-07-25; review cào sau cutoff không tự xuất hiện
  trong frozen release hoặc package nhãn cũ.
- Có thể bảo đảm bằng máy rằng **100% canonical record được hạch toán** qua
  một trong `KEEP`, `KEEP_CLEANED`, `QUARANTINE`, `EXCLUDE_AUTO`, nhờ
  partition-closure validator và checksum. Không thể và không nên cam kết
  100% canonical record đều được gán nhãn ABSA: quarantine, duplicate,
  reserved reference, `ESCALATE` và `REJECT_NON_REVIEW` phải đi nhánh riêng.
- “100%” ở đây là coverage/provenance kỹ thuật, không phải 100% semantic
  accuracy của cleaning hoặc nhãn. Chất lượng học thuật vẫn cần human audit,
  adjudication, IAA và release human-verified.

### Quyết định, giới hạn và next dependency

- **Quyết định đã thực thi:** Không tuyên bố future crawl tự động đi hết
  pipeline; không đưa quarantine/duplicate vào LLM chỉ để đạt tỷ lệ label
  100%; bảo tồn mọi frozen artifact.
- **KẾ HOẠCH CHƯA THỰC THI:** Xây một post-crawl orchestrator fail-closed với
  release ID mới, thực hiện tuần tự: đóng/kiểm manifest → freeze parent →
  validate parent → build curation → validate partition closure → reserve
  leakage groups → prepare eligible tranche → LLM labeling → seal/finalize →
  validate → reconciliation report. Orchestrator phải dừng nếu còn manifest
  `running`, checksum mismatch, record mất/lặp hoặc output version đã tồn tại.
- **Giới hạn vận hành:** Tranche 5.000 cần release-bound archived validator;
  command replay dùng current validator trong README không còn tương thích với
  implementation binding của release cũ và cần được chuẩn hóa trước khi có
  một entry point end-to-end duy nhất.
- **Next dependency:** Chỉ triển khai orchestrator sau khi collection round
  mới kết thúc và người dùng chốt release name/cutoff; labeling phải chờ quyết
  định human-reference reservation và không bao gồm quarantine chưa
  adjudicate.

## TASK-20260728-027 — Chuẩn hóa crawl delta 2026-07-28 qua toàn bộ data flow

- **Trạng thái:** Đã thực thi từ raw audit tới versioned parent, curation,
  leakage-controlled LLM labeling, seal, semantic audit, finalization và
  independent validation. Không sửa/xóa `data/raw/`, không ghi đè base release
  2026-07-25 và không đưa quarantine/duplicate vào ABSA labeling.
- **Mục tiêu:** Chuẩn hóa phần dữ liệu người dùng vừa crawl theo đúng data flow
  đã khóa, dùng cùng generation setup như tranche 8.976 trước đó, đồng thời
  chứng minh record closure ở mỗi ranh giới.
- **Raw input thực tế:** 18 manifest dưới `data/raw/2026-07-28/`, tất cả đã
  đóng; 0 `running`. Chín run `paused_rate_limit`, chín
  `stopped_max_products`; 25 product, 12 transport/error event.

### Raw reconciliation — đã thực thi

- Con số “2k” quan sát khi crawl là candidate progress, không phải 2.000
  accepted unique review. Đếm manifest xác nhận 2.615 candidate được phân
  hoạch kín thành 990 `reviews_written`, 1.056 quality rejection và 569
  cross-run duplicate: 990 + 1.056 + 569 = 2.615.
- Full history dry-run tăng từ base 31.928 lên 32.918 current-policy accepted
  review, chênh đúng 990; còn 21 record policy cũ không được tính.
- Audit độc lập giữa delta parent và frozen parent 31.928 cho 0 `review_id`
  overlap và 0 normalized-text-key overlap. Vì vậy delta được xử lý riêng,
  không chạy lại hoặc thay thế base snapshot.

### Parent release — đã thực thi và VALID

- Build từ riêng `data/raw/2026-07-28/` thành
  `data/releases/lazada_vi_reviews_delta_v1_20260728/`, release ID
  `lazada-vi-substantive_vi_v2-ea14499e4846`.
- Parent có đúng 990 canonical record, 985 primary candidate, 5 manual-review
  candidate, 3 near-duplicate pair/cluster, 10 internal repeated-sentence
  record; label cell đều blank.
- `validate_corpus_release.py` trả `valid`: 18 release file, 72 raw source
  file được xác minh, 0 running manifest. Manifest SHA-256:
  `5a1a2a9360e9828bcd4c7b0721f635c255a46385a7b11b405064224010783099`;
  checksum-ledger SHA-256:
  `15321c81cc2aa17604ac8f64484c114179e196a838a5cb31e0bd485bd310e365`.

### Curation/cleaning — đã thực thi và VALID

- Config đóng băng riêng:
  `configs/cleaning_delta_v1_20260728.json`; giữ nguyên toàn bộ automatic rule
  và threshold của rule version 2.1.2, đổi parent/output sang delta. Các
  `qc_audit_escalations` chứa sample ID human-audit riêng của base 2026-07-25
  được đặt thành mảng rỗng, vì không được chuyển quyết định record-specific
  sang corpus mới.
- Invocation đầu dừng fail-closed trước khi publish vì config copy còn 80
  base-specific QC sample ID không tồn tại trong delta. Temporary build bị
  dọn; không có partial release. Sau khi tách override dataset-specific,
  invocation thứ hai thành công.
- Release:
  `data/releases/lazada_vi_absa_delta_curation_v1_20260728/`, ID
  `lazada-vi-absa-curation-81b2b7522d69f3b9`. Partition:

| Status | Record | Ý nghĩa |
|---|---:|---|
| `KEEP` | 608 | clean-core giữ nguyên |
| `KEEP_CLEANED` | 5 | clean-core sau transformation |
| `QUARANTINE` | 375 | chờ human curation |
| `EXCLUDE_AUTO` | 2 | confirmed punctuation duplicate |
| **Tổng** | **990** | **partition closure 100%** |

- Có 73 transformation, 2 confirmed duplicate alias, 3 near-duplicate pair,
  5 template family. Primary reason lớn nhất trong quarantine là
  `PLATFORM_TEMPLATE` 332; các phần còn lại gồm reward disclosure 29,
  internal repetition 7, non-review suspect 6, post-clean quality 5 và hard
  non-review 1.
- `validate_clean_release_v2.py` trả `VALID`, checksum entries 25. Manifest
  SHA-256:
  `240184bdd231d18bebf45df87abd512d84ecb3263c1adb35c3d4177717224a4b`;
  checksum-ledger SHA-256:
  `b9fa5d15f86589d3355ce02d795aec5aeb32e30997150713fc88db4f7c6866e0`.

### Incremental leakage control — code change và kiểm thử đã thực thi

- Clean-core delta có 613 record. Đối chiếu curated-text SHA-256 với 200
  human-reference và 6.646 reserved leakage-group record trả lần lượt 0 và 0
  overlap; safe frame delta do đó là đúng 613.
- `prepare_ai_annotation_tranche.py` được mở rộng backward-compatible bằng
  chế độ explicit `--incremental-source`, `--reference-release` và
  `--expected-safe-frame`. Chế độ này:
  (1) vẫn replay/checksum original human-reference release để xác minh
  calibration joins; (2) loại leakage trên source mới bằng curated-text hash,
  không chỉ sample ID; (3) yêu cầu expected safe-frame để fail-closed; và
  (4) ghi reference-source binding vào manifest.
- `validate_ai_annotation_tranche.py` được tăng cường kiểm exact-reference và
  reserved-group overlap theo cả sample ID lẫn review-text hash.
- Backward compatibility đạt 20/20 preparation/selection unit test; toàn bộ
  nhóm test liên quan curation, preparation, sealing, finalization và semantic
  audit đạt 83/83. Full repository regression cuối task đạt 194/194 Python
  test.

### LLM labeling — đã thực thi bằng frozen setup

- Prepared package:
  `data/annotations/absa_ai_delta_v1_20260728/`, tranche ID
  `absa-ai-tranche-27dc17ccfd934bf0`, target 613, human-confirmed calibration
  88, accepted prompt calibration 68, diagnostic holdout 20. Reference,
  reservation và prior-tranche overlap đều 0.
- Generation setup khớp tranche 8.976: backend Codex, model
  `gpt-5.6-terra`, reasoning `medium`, batch 20, 8 workers, retry 3,
  max-token 4.096, timeout 420 giây, thinking `auto`; prompt
  `absa-ai-compact-v1.0.0` SHA-256
  `4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`;
  schema SHA-256
  `0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`.
- Diagnostic 20/20, failure 0, một provider call và technical gate `PASS`.
  Primary tạo 613/613 valid record, failure 0. Shell theo dõi chạm timeout
  120 giây sau 160 record nhưng worker process tiếp tục an toàn tới 613; replay
  cùng config dùng 0 provider call, đọc checkpoint và đóng `run_summary.json`.
  Không có record được tạo lại hoặc sửa tay.
- Seal replay trả `SEALED_REPLAY_VALID`: diagnostic 1 attempt; primary 47
  append-only batch attempt; recovered failure marker = 0.
- Terminal distribution: 554 `LABELED`, 46 `ESCALATE`, 13
  `REJECT_NON_REVIEW`; 1.545 mentioned aspect-cell, 72 mixed aspect-cell và
  120 review-level multi-polarity.

### Semantic audit, final release và validator — đã thực thi

- Deterministic difficult-strata semantic audit 60 record, cùng model/reasoning
  nhưng 3 workers: 42 `NO_MATERIAL_ISSUE`, 6 `MINOR_OR_BOUNDARY`, 12 `MAJOR`;
  label mutation = 0. Theo stratum, `MAJOR`: neutral 4, escalate 3, high 3,
  mixed 1, clear 1, reject 0. Đây không phải corpus accuracy vì sample cố ý
  oversample ca khó và auditor vẫn là AI.
- Final:
  `data/annotations/absa_ai_delta_v1_20260728/final/`, đúng 613 record, 275
  human-review queue, trạng thái
  `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`. Independent validator trả
  `VALID`, terminal closure 613, reference overlap 0, reserved-group overlap
  0.
- Final manifest SHA-256:
  `dbae9329ca9fcea1513941e7d128ad1fcd3eeb7cf3baa3a15e491c1484a917ab`;
  checksum-ledger SHA-256:
  `32b6d8fad97dca36d435489776e4ac06939eea3650fbaab998ca2966241bfa84`.
- Machine-readable reconciliation:
  `docs/audits/DELTA_V1_PROCESSING_REPORT_20260728.json`, SHA-256
  `e793b448c2eaa0c5523017add34c6df5f6c6bbab8187a6a7d51ec95a2dc28cba`;
  programmatic checks xác nhận model/prompt/schema match, curation closure và
  annotation terminal closure.

### Quyết định, giới hạn và next dependency

- **Quyết định đã thực thi:** Chỉ label 613 clean-core; bảo tồn 375 quarantine
  và 2 duplicate trong versioned curation ledger; không trộn delta vào base
  release hoặc các pseudo-label release cũ; không gọi số candidate là số
  accepted review.
- **Giới hạn:** 554 `LABELED` vẫn là pseudo-label, không phải gold; 46
  `ESCALATE`, 13 `REJECT_NON_REVIEW`, 275 queue record và 12 semantic-audit
  `MAJOR` cần người kiểm tra. Partition closure 100% không đồng nghĩa semantic
  accuracy 100%.
- **KẾ HOẠCH CHƯA THỰC THI:** Human-check trước 12 `MAJOR`, rồi toàn bộ 275
  queue; human curate phân tầng 375 quarantine; expert adjudicate disagreement;
  phát hành human-verified delta và sau đó mới xây union dataset/split
  group-aware mới. Không sửa in-place pseudo release hiện tại.
- **Next dependency:** Human verification/adjudication; quyết định có phục hồi
  record nào từ 375 quarantine trước khi khóa dataset train/dev/test.

## TASK-20260728-028 — Kiểm kê raw và dữ liệu đã qua labeling

- **Trạng thái:** Đã thực thi audit chỉ đọc manifest/history và bốn final
  pseudo-label release; không thay đổi raw, release hoặc nhãn.
- **Mục tiêu:** Phân biệt raw candidate, accepted unique review, record đã đi
  qua labeling workflow, record có terminal status `LABELED`, và exact-unique
  count toàn corpus.
- **Đầu vào:** 411 crawl manifest dưới `data/raw/`; collector history dry-run;
  final summary/JSONL của tranche 5.000, tranche 8.976, old relabel 9.772 và
  delta 613.

### Raw inventory — đã đo

- 411/411 manifest đã đóng, 0 `running`.
- 72.359 review-candidate encounter được hạch toán thành 32.939 physical
  `reviews_written`, 37.052 quality rejection và 2.368 duplicate:
  32.939 + 37.052 + 2.368 = 72.359.
- Trong 32.939 physical written row, collector history công nhận **32.918
  accepted unique current-policy review**; 21 record policy cũ/không hợp lệ
  bị loại khỏi current target.
- Hai parent release versioned đóng đúng toàn bộ current-policy accepted:
  base 31.928 + delta 990 = 32.918.

### Crawled-data curation và labeling inventory

- Curation union base + delta:
  21.235 clean-core, 11.541 quarantine và 142 confirmed duplicate exclusion;
  21.235 + 11.541 + 142 = 32.918.
- Trong 21.235 clean-core, 6.646 human-reference leakage-group record được
  reserve. Phần safe-frame đã đi qua LLM workflow là **14.589**:
  13.976 base + 613 delta.
- Terminal status của 14.589 crawled pseudo-label record:
  13.381 `LABELED`, 982 `ESCALATE`, 226 `REJECT_NON_REVIEW`.

### Tổng khi cộng old dataset relabel

- Old canonical unique đã đi qua workflow: 9.772, gồm 8.553 `LABELED`, 950
  `ESCALATE`, 269 `REJECT_NON_REVIEW`. Không cộng projection 10.105 vì 333
  source-row alias không phải review được label độc lập.
- Tổng bốn canonical pseudo-label release:
  **24.361 record đã đi qua labeling workflow** =
  13.976 base + 613 delta + 9.772 old.
- Terminal status tổng: **21.934 `LABELED`**, 1.932 `ESCALATE`, 495
  `REJECT_NON_REVIEW`; tổng đóng đúng 24.361.
- Union review-text SHA-256 có **24.360 exact-unique text**, thấp hơn một vì
  old 9.772 và tranche base 8.976 có đúng một overlap. Nếu chỉ đếm terminal
  `LABELED`, có 21.933 exact-unique text trên 21.934 record.
- Delta 613 không overlap với base hay old; tranche 5.000 và 8.976 không giao
  nhau.

### Quyết định, giới hạn và next dependency

- **Quyết định:** Khi báo số dataset, dùng 32.918 cho accepted unique raw crawl;
  dùng 14.589 cho crawled records đã đi qua LLM workflow; dùng 24.361 nếu nói
  rõ đã cộng old relabel; không gọi toàn bộ 24.361 là human-labeled hoặc gold.
- **Giới hạn:** `LABELED` vẫn là AI pseudo-label pending human verification.
  Candidate count 72.359 không phải dataset size; physical written 32.939
  cũng không phải current canonical size.
- **Next dependency:** Human verify 275 delta queue và các queue base/old,
  adjudicate `ESCALATE`, rồi mới phát hành count human-verified/gold riêng.

## TASK-20260728-029 — Kiểm kê phần raw chưa được labeling và nguyên nhân

- **Trạng thái:** Đã thực thi audit chỉ đọc; 411 manifest đã đóng, 0
  `running`; không thay đổi raw/release/nhãn.
- **Mục tiêu:** Phân biệt record chưa từng được gửi qua LLM với record đã qua
  LLM nhưng chưa có terminal status `LABELED`.
- **Đầu vào:** Union curation base + delta; ba final pseudo-label release của
  crawled data (5.000, 8.976, 613); đếm trực tiếp primary reason trong hai
  `quarantine.jsonl`.

### Chưa từng qua LLM labeling

- Trên 32.918 accepted unique raw crawl, 14.589 đã đi qua LLM workflow.
  **18.329 record chưa được gửi qua LLM**, phân rã kín:

| Nhánh | Record | Lý do |
|---|---:|---|
| Quarantine | 11.541 | Heuristic/risk cần human curation trước |
| Human-reference leakage-group reserve | 6.646 | Bảo vệ tính độc lập của 200 reference review và các group liên quan |
| Confirmed duplicate exclusion | 142 | Alias/trùng nội dung, không label như observation độc lập |
| **Tổng** | **18.329** | **32.918 − 14.589** |

- Primary reason trong 11.541 quarantine: 10.206 `PLATFORM_TEMPLATE`, 690
  `REWARD_DISCLOSURE`, 208 `POST_CLEAN_QUALITY`, 161
  `INTERNAL_REPETITION`, 157 `NONREVIEW_SUSPECT`, 60 `HARD_NONREVIEW`, 34
  `NEAR_DUPLICATE`, 24 `PRIVACY`, 1 `DUPLICATE` mơ hồ. Tổng đúng 11.541.
- Quarantine không đồng nghĩa toàn bộ là rác; đặc biệt 10.206 template là
  heuristic bảo thủ và cần human recover/reject decision. Reserved 6.646 là
  clean-core nhưng cố ý không pseudo-label để tránh evaluation leakage.

### Đã qua LLM nhưng chưa đạt `LABELED`

- Trong 14.589 crawled record đã qua workflow: 13.381 `LABELED`, 982
  `ESCALATE`, 226 `REJECT_NON_REVIEW`.
- Do đó nếu “chưa labeling” được hiểu là **chưa có nhãn dùng ngay**, con số là
  **19.537** = 18.329 chưa qua LLM + 982 cần adjudication + 226 non-review bị
  reject; tương đương 32.918 − 13.381.
- `ESCALATE` có provisional annotation/evidence nhưng còn ambiguity,
  aspect-boundary hoặc semantic uncertainty, không được coi như verified
  label. `REJECT_NON_REVIEW` đã được model xử lý nhưng cố ý không có vector
  ABSA hợp lệ để train như review.
- Old canonical 9.772 đã được gửi qua workflow đầy đủ; riêng old có 8.553
  `LABELED`, 950 `ESCALATE`, 269 `REJECT_NON_REVIEW`. Vì vậy old không làm
  tăng số “chưa từng qua LLM”, nhưng vẫn còn 1.219 record chưa đạt
  `LABELED`.

### Quyết định, giới hạn và next dependency

- **Quyết định:** Báo hai con số riêng: 18.329 chưa qua LLM và 19.537 chưa có
  terminal `LABELED`; không gộp `ESCALATE` với unlabeled mà không giải thích.
- **Giới hạn:** `LABELED` hiện vẫn là AI pseudo-label pending human
  verification, không phải human-gold hoặc train-ready mặc định.
- **Next dependency:** Human curate 11.541 quarantine (ưu tiên stratified
  template audit), giữ 6.646 reserve ngoài evaluation-trained model, và
  adjudicate 982 crawled `ESCALATE`; duplicate 142 tiếp tục giữ trong ledger,
  không phục hồi thành observation độc lập.

## TASK-20260728-030 — Data-flow processing và pseudo-label toàn bộ quarantine

- **Trạng thái:** ĐÃ THỰC THI và VALID. Đã xử lý đủ 11.541/11.541 record từ
  hai partition quarantine versioned; không sửa/xóa `data/raw/`, không thay
  `curation.status`, không nhập các record này vào clean core hoặc train set.
- **Mục tiêu:** Theo quyết định mới của người dùng sau khi đọc review, đưa cả
  hai quarantine partition qua đúng annotation workflow đã dùng trước đó,
  nhưng vẫn bảo toàn provenance quarantine, fail-closed validation, review
  queue và giới hạn pseudo-label.
- **Đầu vào đã dùng:**
  `data/releases/lazada_vi_absa_curation_v2_1_2_20260725/quarantine.jsonl`
  (11.166 record) và
  `data/releases/lazada_vi_absa_delta_curation_v1_20260728/quarantine.jsonl`
  (375 record). Union có 11.541 sample ID và 11.541 curated-text SHA-256 duy
  nhất; cross-partition overlap theo cả hai khóa đều 0.

### Code/data-flow thay đổi — đã thực thi và kiểm thử

- `prepare_ai_annotation_tranche.py` được mở rộng backward-compatible bằng
  `--source-partition {clean_core,quarantine}`. Chế độ quarantine chỉ được
  phép cùng `--incremental-source`, join trực tiếp `quarantine.jsonl`, yêu cầu
  curation status đúng `QUARANTINE`, tạo annotation ID tiền tố `aiq-`, và ghi
  `source_mode=QUARANTINE_PARTITION`/source artifact hash vào manifest. Chế độ
  clean-core cũ và các legacy field trong manifest vẫn được giữ.
- `validate_ai_annotation_tranche.py` được mở rộng để xác minh generic
  `records_path`/SHA/count, bind `source_partition` với đúng source artifact,
  và fail nếu quarantine output không join kín partition hoặc mất status
  `QUARANTINE`.
- `run_ai_semantic_audit.py` có tùy chọn explicit
  `--allow-stratum-backfill` cho package nhỏ: lấy tối đa 10 record ở từng
  difficult stratum rồi bù deterministic SHA-256 tới đúng 60 ID duy nhất.
  Backfill được ghi thành stratum riêng; mặc định strict sampling cũ không
  đổi. Hai unit test mới xác minh strict failure và deterministic backfill.
- `build_quarantine_labeling_report.py` được thêm để programmatically kiểm
  source-to-final closure, config drift, cross-package/prior-release overlap,
  terminal closure và tạo machine-readable reconciliation.
- **Kiểm thử đã chạy:** full Python regression đạt **196/196**; compile check
  cho các script thay đổi đạt. Đây là software regression, không phải semantic
  accuracy.

### Preparation, leakage control và diagnostic — đã thực thi

| Package | Tranche ID | Target | Diagnostic | Leakage/overlap |
|---|---|---:|---:|---|
| `absa_ai_quarantine_base_11166_v1_20260728` | `absa-ai-tranche-fc86047482d219c0` | 11.166 | 20/20 valid, gate `PASS` | reference 0; reserved group 0; prior tranche 0 |
| `absa_ai_quarantine_delta_375_v1_20260728` | `absa-ai-tranche-037f1c7c14166722` | 375 | 20/20 valid, gate `PASS` | reference 0; reserved group 0; prior tranche 0 |

- Mỗi package dùng lại 88 human-confirmed calibration record: 68
  `CALIBRATION_ACCEPT`, 20 diagnostic holdout; không dùng 20 holdout làm prompt
  example. Base diagnostic trả 20 `LABELED`; delta trả 18 `LABELED`, 2
  `ESCALATE`. Cả hai gate chỉ chứng minh technical alignment với calibration,
  không chứng minh human-gold accuracy.
- Frozen generation setup khớp các tranche trước: backend Codex, model
  `gpt-5.6-terra`, reasoning `medium`, batch 20, 8 workers, retry 3,
  max-token 4.096, timeout 420 giây, thinking `auto`/serialized `null`;
  prompt `absa-ai-compact-v1.0.0` SHA-256
  `4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`;
  output schema SHA-256
  `0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`.

### Primary labeling và replay — đã thực thi

- Delta primary hoàn tất 375/375 trong một invocation, 23 provider
  batch-attempt, thiếu ID = 0.
- Base lượt đầu tạo 11.164/11.166 valid record. Hai record bị fail-closed:
  một evidence quote rỗng và một absent aspect còn evidence. Không record lỗi
  nào được tính hoặc sửa tay. Replay đúng frozen config chỉ gọi model hai lần,
  phục hồi đủ 11.166/11.166 và đóng summary `COMPLETED`.
- Seal replay base: 719 attempt (`VALID` 559, `PARTIAL_VALID` 160), 11.166
  record bind tới valid attempt duy nhất, 6 normalization repair, 2 recovered
  failure marker, 0 unrecovered failure. Delta seal: 23 attempt, 0 recovered
  marker, 0 unrecovered failure.
- Terminal annotation distribution:

| Nguồn quarantine | `LABELED` | `ESCALATE` | `REJECT_NON_REVIEW` | Tổng |
|---|---:|---:|---:|---:|
| Base | 10.119 | 661 | 386 | 11.166 |
| Delta | 348 | 20 | 7 | 375 |
| **Union** | **10.467** | **681** | **393** | **11.541** |

- Union có 22.282 mentioned aspect-cell, 138 mixed aspect-cell và 243 review
  multi-polarity. `LABELED` ở đây chỉ nghĩa là output AI hợp schema/evidence
  validator; không đồng nghĩa nhãn đã được người xác minh.

### Semantic audit — đã thực thi, không sửa nhãn

- Base audit chọn deterministic 10 record cho mỗi tầng `reject`, `escalate`,
  `neutral`, `mixed`, `high`, `clear`: 40 `NO_MATERIAL_ISSUE`, 6
  `MINOR_OR_BOUNDARY`, 14 `MAJOR`.
- Delta không có đủ 10 candidate trong mọi tầng (chỉ 7 reject và 3 mixed
  candidate sau selection order), nên dùng explicit deterministic backfill
  tới đủ 60 unique record: 54 `NO_MATERIAL_ISSUE`, 1
  `MINOR_OR_BOUNDARY`, 5 `MAJOR`; 12 record thuộc `backfill`.
- Tổng hai audit là 120 record, label mutation = 0. Vì sample cố ý oversample
  ca khó, và auditor vẫn là AI cùng model family, các tỷ lệ severity **không
  được báo như corpus accuracy, human accuracy hoặc IAA**.

### Final release và independent validation — đã thực thi

| Final release | Record | Human-review queue | Validator | Manifest SHA-256 | Checksum-ledger SHA-256 |
|---|---:|---:|---|---|---|
| `data/annotations/absa_ai_quarantine_base_11166_v1_20260728/final/` | 11.166 | 2.638 | `VALID` | `62b6c55ace3c350ffbc3bc52a83c264fd0a3fb8db9a6302d7c9997eb33621f28` | `d0bd0557ac53485c5d15b2c17fa09c9ddb9ab61c3f0c28461a7cfb5d1d72be0c` |
| `data/annotations/absa_ai_quarantine_delta_375_v1_20260728/final/` | 375 | 94 | `VALID` | `433f980e9bbd564bed40b58da9684c7235e659429d622a2660eb4969b5224944` | `86730a6b64fb004d836b93bcdf104781d7e974263295b15746ad8b4cd62f24ff` |

- Cả hai validator trả human-reference overlap 0, reserved-group overlap 0 và
  terminal closure 100%. Union final có 11.541 unique sample ID, 11.541 unique
  review-text hash, cross-package overlap 0 và overlap với bốn final
  pseudo-label release trước = 0.
- Machine-readable report:
  `docs/audits/QUARANTINE_LABELING_REPORT_20260728.json`, status `VALID`,
  SHA-256
  `f2badeddaa50dfc2b502ce6321e9838b9e3ed1bf12423647d0f4b998abd87507`.

### Reconciliation sau task — số đã đo

- Trên 32.918 accepted unique crawled review, số đã đi qua LLM workflow tăng
  từ 14.589 lên **26.130**. Phần còn cố ý không pseudo-label là 6.646
  reserved-reference-group và 142 confirmed duplicate:
  26.130 + 6.646 + 142 = 32.918.
- Terminal crawled distribution sau task: 23.848 `LABELED`, 1.663
  `ESCALATE`, 619 `REJECT_NON_REVIEW`; tổng đúng 26.130.
- Nếu cộng old canonical relabel 9.772: có 35.902 record qua workflow, gồm
  32.401 `LABELED`, 2.613 `ESCALATE`, 888 `REJECT_NON_REVIEW`. Review-text
  exact-unique = 35.901 vì đã biết đúng một overlap old/base từ audit trước.

### Quyết định, giới hạn và next dependency

- **Quyết định đã thực thi:** Label toàn bộ quarantine theo yêu cầu mới nhưng
  giữ `source.curation_status=QUARANTINE`; publish thành hai pseudo-label
  release riêng; không đổi quyết định curation, không nhập clean core, không
  loại im lặng `ESCALATE`/`REJECT_NON_REVIEW`.
- **Giới hạn:** Quarantine được chọn bởi heuristic nên có selection bias;
  10.467 `LABELED` chưa phải train-ready/human-gold; 681 `ESCALATE`, 393
  `REJECT_NON_REVIEW`, 2.732 deterministic/risk queue record và 19
  semantic-audit `MAJOR` cần human review. Queue count không tự động đồng
  nghĩa mọi semantic-audit `MAJOR` đã được người xử lý.
- **KẾ HOẠCH CHƯA THỰC THI:** Human-check 19 `MAJOR` trước, sau đó toàn bộ
  `ESCALATE` và `REJECT_NON_REVIEW`, rồi phần còn lại của hai review queue;
  adjudicate disagreement và chỉ sau đó mới phát hành partition
  human-verified hoặc quyết định record nào được phục hồi vào train corpus.
- **Next dependency:** Human verification/adjudication và một versioned
  decision ledger mới; tuyệt đối không sửa in-place hai final pseudo-label
  release hiện tại.

## TASK-20260728-031 — Khóa Q1 logical dataset snapshot v1

- **Trạng thái:** ĐÃ THỰC THI và independent validator `VALID`.
- **Mục tiêu:** Khóa chính xác trạng thái collection, curation, human-reference
  và pseudo-label hiện tại trước khi làm Q1 corpus audit; ngăn việc một file
  nguồn bị thay âm thầm trong các bước tiếp theo.
- **Đầu vào:** Toàn bộ `data/raw/`; hai accepted canonical release; hai
  curation release; human-reference release; sáu final pseudo-label release
  (clean base 5.000 + 8.976, clean delta 613, quarantine 11.166 + 375 và old
  canonical 9.772); collector/cleaning config, guideline V2 và quarantine
  reconciliation report.
- **Phương pháp/code đã thực thi:**
  `scripts/build_q1_dataset_snapshot.py` hash SHA-256 từng raw file và bind
  mỗi published release bằng manifest + checksum ledger; build atomically và
  refuse overwrite. `scripts/validate_q1_dataset_snapshot.py` độc lập kiểm
  closed artifact inventory, checksum ledger, raw membership/hash, release
  binding và các phương trình count closure.
- **Output:** `data/releases/q1_dataset_snapshot_v1_20260728/`, snapshot ID
  `q1-dataset-snapshot-03caeced3c2f421a`; 1.645 raw file, 53.195.482 byte,
  11 release binding. Manifest SHA-256
  `fa436e457e02d74c3e7f0c93e218aaec01a2f4adc0ebe0f47ca02e38377ee74c`;
  checksum-ledger SHA-256
  `9e977a1f72b80a6d45de035431709477664e4ecdcbe7fbb81b589c9b1aa8dd10`.
- **Kết quả đo:** Validator xác nhận raw/release không drift và closure:
  26.130 crawled record qua LLM + 6.646 reserved group + 142 duplicate =
  32.918 accepted unique crawled review.
- **Quyết định:** Đây là logical checksum freeze, không copy/khóa quyền ghi
  OS và không biến pseudo-label thành human-gold. Các source payload tiếp tục
  bất biến; mọi output mới phải versioned và tham chiếu snapshot ID.
- **Giới hạn:** Snapshot chỉ phát hiện drift khi chạy validator; không phải
  cơ chế backup hoặc filesystem write protection.
- **Next dependency:** Q1 Collection & Corpus Audit chạy trên đúng snapshot
  này; sampling frame cho gold data chưa được tạo ở task này.

## TASK-20260728-032 — Q1 Collection & Corpus Audit v1

- **Trạng thái:** ĐÃ THỰC THI; audit release và independent validator đều
  `VALID`.
- **Mục tiêu:** Mô tả định lượng collection frame và các selection effect
  trước khi thiết kế human-gold sampling; tách rõ crawled corpus với old
  historical corpus và không dùng pseudo-label như ground truth.
- **Đầu vào:** Snapshot `q1-dataset-snapshot-03caeced3c2f421a`; 411 raw
  manifest/rejection ledger; accepted canonical base + delta; curation base +
  delta; năm crawled pseudo-label final và old final được báo riêng.
- **Code/phương pháp đã thực thi:**
  `scripts/run_q1_collection_corpus_audit.py` kiểm candidate-accounting,
  membership closure và đo run status, rejection reason, category/rating/query,
  transport, độ dài, quality, review-time coverage, product concentration,
  curation, annotation status, aspect/polarity và coverage theo stratum.
  Script phát hành atomic JSON/Markdown + 14 CSV table + provenance/checksum.
  `scripts/validate_q1_collection_corpus_audit.py` độc lập kiểm snapshot
  binding, closed artifact inventory, checksum và count equations.

### Collection process — số đã đo

- Collection window: `2026-07-23T12:34:20.849523+00:00` đến
  `2026-07-28T12:36:50.216530+00:00`; 411/411 run đã đóng, open/running = 0.
- Run status: 206 `paused_rate_limit`, 117 `stopped_max_products`, 72
  `failed`, 7 `completed_target`, 5 `interrupted_external`, 3
  `completed_with_shortfall`, 1 `completed`. Có 432 error event được ghi ở
  run-level; không bị diễn giải thành 432 record lỗi độc lập.
- Candidate accounting đóng kín:
  **72.359 = 32.939 written + 37.052 quality-rejected + 2.368
  deduplicated**. Current-policy canonical accepted là 32.918; 21 physical
  written row thuộc policy cũ/không còn trong canonical target.
- Quality rejection là multi-reason: 35.030 `TOO_SHORT_CHARS`, 27.577
  `TOO_FEW_WORDS`, 22.792 `LOW_QUALITY_SCORE`, 17.766
  `TOO_FEW_MEANINGFUL_WORDS`, 4.902 `INSUFFICIENT_VIETNAMESE_SIGNAL`, 592
  low lexical diversity, 130 foreign-script dominant và 40 suspect encoding.
  Vì một review có thể có nhiều reason nên không cộng các reason thành số
  review bị loại.

### Accepted corpus — số đã đo

- 32.918 unique accepted record, 974 product, 27 stored query. Median 117 ký
  tự, 26 từ và quality score 0,7547. Verified-purchase share 78,49%; has-image
  share 80,03%.
- Review date parse được 81,98%; observed range 2017-01-18 đến 2026-06-25.
  18,02% không parse được nên temporal distribution phải báo missingness.
- Rating cực lệch: 31.355 rating 5 (**95,25%**), 673 rating 4 (2,04%),
  452 rating 1 (1,37%), 286 rating 3 (0,87%), 152 rating 2 (0,46%).
- Category lớn nhất `food_beverage` 6.455 (19,61%); tiếp theo
  `beauty_personal_care` 5.207, `electronics` 4.686, `fashion` 4.254,
  `home_appliances` 3.586, `home_living` 3.034, `mother_baby` 3.029,
  `sports_outdoors` 2.346. Có 315 missing category và 6 giá trị legacy
  `Men shoes`; không được âm thầm map khi chưa có quyết định.
- Transport: 28.214 `requests_cookie` (85,71%), 4.693 `selenium_dom`
  (14,26%), 11 legacy `requests` (0,03%).
- Product concentration không quá phụ thuộc một sản phẩm nhưng vẫn phải
  group-split: top-1 share 1,34%, top-10 11,37%, HHI 0,002550, Gini 0,4040.

### Curation và labeling coverage — số đã đo

- Curation closure 32.918 = 21.235 clean-core + 11.541 quarantine + 142
  duplicate exclusion; quarantine share 35,06%.
- Crawled LLM workflow coverage 26.130/32.918 = **79,38%**; 23.848
  `LABELED`, 1.663 `ESCALATE`, 619 `REJECT_NON_REVIEW`.
- Coverage không đồng đều: theo rating, workflow coverage từ 67,11% (rating
  2) tới 79,61% (rating 5); theo category, 72,59% `home_appliances` tới
  86,34% `fashion` trong tám mapped categories. Sáu `Men shoes` legacy record
  chưa qua workflow.
- Review-length median: accepted 117, clean-core 116, quarantine 112,
  LLM-workflow 114, terminal `LABELED` 113, old canonical 91 ký tự. Đây là
  corpus đã chủ ý chọn review dài/chất lượng, không đại diện toàn bộ review
  ngắn trên Lazada.
- Old canonical 9.772 chỉ được báo như historical external corpus, không cộng
  vào collection distribution của crawl hiện tại.

### Output, quyết định và giới hạn

- Audit release:
  `docs/audits/q1_collection_corpus_audit_v1_20260728/`, audit ID
  `q1-corpus-audit-0ecc6d2beebb03f7`. Manifest SHA-256
  `a62aacbc92cd2b94b7402b7f9d63cbd809c5f45d325c0e9e005f3ca0543c6d88`;
  checksum-ledger
  `40b614936369ceb1e6bbdef9721ecd74d8413b26161c17c9798fbf3fa033de41`;
  report SHA-256
  `b6d7e5b9ab12fc477756f571f7273f77ea7975405a054e82d84257088ff49455`.
- **Quyết định:** Không random-sample gold/test trực tiếp từ corpus 95,25%
  rating 5. Gold frame phải joint-stratify category, rating, transport,
  curation origin, annotation status, aspect/polarity rarity và product group.
- **Giới hạn:** Single-platform; không có demographic/geographic/full-seller
  population frame; query, availability, rate limit, transport fallback và
  long-review filter đều tạo selection effect. Audit mô tả frozen sample,
  không chứng minh population representativeness hoặc label accuracy.
- **Next dependency:** Thiết kế và khóa human-gold/IAA sampling frame dựa trên
  các table audit; chưa gán một review nào vào train/dev/test ở task này.

## TASK-20260728-033 — Khóa Q1 human-gold & IAA sampling plan v1

- **Trạng thái:** ĐÃ THỰC THI preflight; sampling-plan release
  `VALID_FEASIBLE`, independent validator `VALID`. Chưa tạo assignment hoặc
  nhãn người ở task này.
- **Mục tiêu:** Định trước gold/IAA design trước khi xem kết quả human label;
  tách blind gold khỏi AI-assisted review và loại calibration leakage ở mức
  product/exact/near-duplicate/template group.
- **Đầu vào:** Snapshot `q1-dataset-snapshot-03caeced3c2f421a`, corpus audit
  `q1-corpus-audit-0ecc6d2beebb03f7`, union 26.130 crawled pseudo-label
  record, 200 calibration reference record, 6.646 prior reserved-group record
  và 300 semantic-audit ledger record từ năm crawled packages.
- **Config/code đã thực thi:**
  `configs/q1_human_gold_sampling_v1.json` khóa target/marginal quota,
  double-blind/adjudication/split policy.
  `scripts/design_q1_human_gold_sampling.py` dựng lại leakage component bằng
  product ID, exact curated-text SHA, near-duplicate representative và
  template-family ID; preflight mọi quota và phát hành report/table/checksum.
  `scripts/validate_q1_human_gold_sampling_plan.py` độc lập kiểm source/config
  binding, closed inventory và feasibility gates.

### Eligibility và leakage — số đã đo

- Pseudo frame ban đầu 26.130. Existing 200 human-reference chỉ dùng
  calibration, không dùng final evaluation; toàn bộ 6.646 prior reserved
  record tiếp tục bị loại.
- Expanded grouping phát hiện 146 component liên thông với prior calibration
  reserve; để fail-safe, **10.577** pseudo record trong các component này bị
  loại khỏi gold frame, dù sample ID không trùng trực tiếp.
- Gold-eligible frame còn **15.553 unique record**, 586 leakage component;
  group size min 1, median 21, mean 26,54, max 130. Old corpus và confirmed
  duplicate included = 0.
- Có 53 semantic-audit `MAJOR` trên toàn crawled corpus; chỉ 23 nằm trong
  leakage-independent frame. 30 record còn lại không bị bỏ: chúng thuộc
  human-review priority nhưng không được dùng independent gold test.

### Sampling design đã khóa

- **Target 1.200 unique review, hai annotator cùng đánh blind toàn bộ** =
  2.400 annotation task trước adjudication. AI suggestion/evidence bị ẩn;
  role A/B dùng opaque ID và thứ tự độc lập.
- `CORE_BALANCED` 800 record: tám mapped category, mỗi category 100; tối
  thiểu mỗi category 10 rating 1/2 và 10 rating 3/4; toàn panel tối thiểu 120
  Selenium-DOM, 320 clean-origin và 320 quarantine-origin. Feasibility của cả
  tám category `PASS`.
- `CHALLENGE_DIAGNOSTIC` 400 record, ordered disjoint quotas: 23 independent
  semantic-major, 170 `ESCALATE`, 90 `REJECT_NON_REVIEW`, 55 mixed, 35
  neutral và 27 negative/rare-aspect. Availability tương ứng:
  23, 998, 331, 1.028, 1.134 và 1.877; mọi gate `PASS`.
- Package builder sau task này được giới hạn tối đa 240 leakage component,
  tối đa 8 core record và 12 total record mỗi component. Toàn component đã
  chọn phải bị reserve khỏi future training.
- Sau adjudication: core được group-split 300 dev + 500 test; challenge 400
  báo riêng, không aggregate như prevalence. IAA phải tính trước adjudication:
  status exact, aspect mention F1, per-aspect polarity Cohen's kappa,
  Krippendorff's alpha và confusion matrices.

### Output, quyết định và giới hạn

- Plan release:
  `docs/audits/q1_human_gold_sampling_plan_v1_20260728/`, plan ID
  `q1-human-gold-plan-a7db74c454ac54b0`. Manifest SHA-256
  `57d4f89091660ada83636426594854c74beeea07c83e208b43d2b7d529646431`;
  checksum-ledger
  `0d0bbec97ac19d6813cd2be28dfca2625cc6842265b5309f76cb8bb1eaa494fc`;
  report
  `108636dbf69efd8dc97d7d0eb5f120e012afe42e377439e935960b2870c82419`.
- **Quyết định:** Đây là balanced evaluation design, không phải prevalence
  sample. Blind gold phải hoàn thành trước khi cùng sample được mở trong
  AI-review mode. `CORE_BALANCED` dùng dev/test; challenge chỉ diagnostic.
- **Giới hạn:** Feasibility không thay thế hai người thật và expert
  adjudication. Rating/source selection bias vẫn tồn tại; không báo 800 core
  như phân phối tự nhiên của Lazada.
- **Next dependency:** Deterministic package builder phải chọn đúng 1.200
  record theo plan, khóa assignment blindness, independent role order và
  selected-group reservation trước khi UI bắt đầu annotation.

## TASK-20260728-034 — Audit và khóa thiết kế Q1 annotation workbench

- **Trạng thái:** ĐÃ THỰC THI phần audit/thiết kế; chưa triển khai code mode
  mới và chưa tạo human annotation.
- **Mục tiêu:** Quyết định rõ dùng một hay nhiều UI; xác định separation
  giữa blind gold, AI-assisted review và adjudication trước khi tạo package
  1.200 review.
- **Đầu vào đã đọc:** `human_annotation_ui/index.html`, `app.js`,
  `styles.css`, `annotation_core.mjs`, `common.py`, `serve.py`,
  `ai_review.py`, `validate_export.py`, hai start script, README và unit
  test; sampling plan `q1-human-gold-plan-a7db74c454ac54b0`.
- **Audit đã thực thi:** Xác nhận UI hiện có đã strict-validate assignment,
  hash text/payload, không đưa sampling/source metadata vào blind assignment,
  autosave IndexedDB theo assignment, revision/export ledger, exact evidence,
  local-only CSP server và Python export validator. AI suggestion đã có mode
  banner và endpoint riêng.
- **Thiếu sót đo/xác định:** Server còn suy mode từ `--suggestions`; role chỉ
  A/B; mode chưa bind vào workspace/export; chưa có A/B comparator,
  disagreement/adjudication ledger; start script và copy còn hard-code bộ
  200; chưa có end-to-end persistence/mode-isolation smoke test.
- **Quyết định đã khóa:** Dùng **một annotation workbench với ba mode**
  `BLINDED_INDEPENDENT_ANNOTATION`,
  `AI_ASSISTED_HUMAN_VERIFICATION`, `EXPERT_ADJUDICATION`; cả ba dùng chung
  Guideline V2/semantic schema nhưng khác input visibility, persistence key,
  export contract và validator gate. Không tạo hai hệ thống nhãn song song.
- **Stage gate:** Hai blind FINAL phải được validate/freeze; IAA phải được
  publish trước adjudication; AI suggestion không được mở cho gold sample
  trước khi blind A/B đóng; core chỉ group-split 300 dev + 500 test sau
  expert-final, challenge 400 báo riêng.
- **Output:** `docs/audits/Q1_ANNOTATION_WORKBENCH_DESIGN_V1.md`, gồm
  architecture, audit, stage table, UI IA, persistence/audit contract,
  validation strategy và ranh giới executed/planned.
- **Giới hạn:** Đây là design freeze, không phải bằng chứng UI đã được sửa
  hoặc human đã gán. Không có IAA/adjudication result vì chưa có hai human
  FINAL.
- **Next dependency:** Build deterministic gold package 1.200 và khóa
  selected-group reservation; sau đó mới tích hợp package/mode contract vào
  workbench và chạy smoke test.

## TASK-20260728-035 — Fail-closed package-build preflight: phát hiện quota Selenium bất khả thi

- **Trạng thái:** ĐÃ THỰC THI; package build bị dừng có chủ ý, không có
  partial/final package được publish.
- **Mục tiêu:** Chạy deterministic selector theo sampling plan v1 và xác minh
  đồng thời mọi marginal quota với leakage-group cap, thay vì chỉ kiểm
  aggregate record availability.
- **Đầu vào:** Frame 15.553 record của plan
  `q1-human-gold-plan-a7db74c454ac54b0`; config v1; group graph theo product,
  exact/near duplicate và template family; trần 8 core/12 total record mỗi
  group.
- **Code/phương pháp đã thực thi:** Tạo
  `scripts/build_q1_human_gold_package.py`; deterministic challenge selection,
  core marginal selection, role A/B opaque assignment builder và group
  reservation builder. Script được compile, sau đó chạy thật trên frozen
  frame và fail tại core Selenium margin trước khi publish output.
- **Kết quả đo mới:** 231 Selenium-eligible record chỉ nằm trong 9
  leakage component: beauty 41/2 group/cap 16; electronics 9/2/cap 9; home
  appliances 85/2/cap 16; home living 16/2/cap 13; mother & baby 80/1/cap 8.
  Tổng theoretical core capacity dưới trần 8/group chỉ **62**, nhỏ hơn quota
  v1 là 120. Sau provisional challenge selection, một số scarce group còn bị
  dùng hết total capacity, khiến selector dừng tại beauty còn thiếu 4
  Selenium record.
- **Quyết định:** Không nới trần group và không tự ý bỏ quota. Sampling plan
  v1 được giữ nguyên như audit trail nhưng bị **supersede trước khi chọn
  sample**. Sẽ phát hành config/plan v1.1 với group-aware feasibility và quota
  Selenium thấp hơn nhưng có headroom; selection seed mới để phân biệt.
- **Output:** Source builder đã được tạo nhưng chưa sinh
  `data/annotations/q1_human_gold_1200_v1_20260728/`; raw và mọi release cũ
  không bị sửa.
- **Giới hạn:** Capacity 62 là upper bound riêng cho Selenium/core dưới group
  cap, chưa bảo đảm các quota rating/curation/challenge đồng thời khả thi.
  Plan v1.1 và builder thực tế vẫn phải đóng toàn bộ constraints.
- **Next dependency:** Phát hành và independent-validate sampling plan v1.1,
  rồi chạy lại builder; chỉ package qua validator mới được bàn giao cho UI.

## TASK-20260728-036 — Phát hành corrected Q1 human-gold sampling plan v1.1

- **Trạng thái:** ĐÃ THỰC THI; plan `VALID_FEASIBLE`, independent validator
  `VALID`.
- **Mục tiêu:** Sửa đúng infeasibility của plan v1 mà không thay đổi target
  1.200, double-blind design, challenge quota hoặc leakage-group cap.
- **Đầu vào:** Snapshot/corpus audit/frame/group graph như TASK-033/035; upper
  bound Selenium dưới trần 8 core/group = 62.
- **Config/code đã thực thi:** Tạo
  `configs/q1_human_gold_sampling_v1_1.json` với selection seed mới,
  `supersedes_plan_id=q1-human-gold-plan-a7db74c454ac54b0`, Selenium minimum
  50. `scripts/design_q1_human_gold_sampling.py` được nâng cấp đo
  group-aware capacity, không chỉ aggregate availability. Tạo independent
  validator `scripts/validate_q1_human_gold_sampling_plan_v1_1.py`.
- **Kết quả đo:** Eligibility giữ nguyên 15.553 record; 231 Selenium record
  thuộc 9 group; group-aware capacity 62; required 50, headroom 12. Mọi core
  category, ordered challenge quota và global gate khác tiếp tục `PASS`.
- **Output:** 
  `docs/audits/q1_human_gold_sampling_plan_v1_1_20260728/`, plan ID
  `q1-human-gold-plan-f8ba30c091efa284`; manifest SHA-256
  `f2ca4dea5e768a63d37972194ee92643b1fec713319f9b4651e3ea37b73a48d6`;
  checksum-ledger
  `6717725a7a827eb0b543fa8bda00d2a432c9ee0e8080afeda5788ba1fda018e3`;
  report
  `bc809c19b5f8e896db7c8c18f79d3ca689ea373d9bd0e5f20b52e35f460a1c22`.
- **Quyết định:** Giữ max 8 core/max 12 total mỗi leakage component vì
  chống product/group dominance quan trọng hơn việc ép 120 Selenium. Quota
  50 vẫn oversample transport thiểu số nhưng nằm dưới upper bound với
  headroom; plan v1 được lưu audit trail, không xóa/ghi đè.
- **Giới hạn:** Group-aware upper bound riêng lẻ chưa chứng minh simultaneous
  feasibility của toàn bộ constraint. Deterministic builder và package
  validator vẫn phải chạy thật; không có human label ở task này.
- **Next dependency:** Chạy lại package builder theo đúng plan ID/seed v1.1,
  bảo vệ scarce Selenium group khỏi challenge fill, rồi independent-validate
  public blindness và reservation closure.

## TASK-20260728-037 — Build và independent-validate Q1 human-gold package 1.200

- **Trạng thái:** ĐÃ THỰC THI; package
  `BUILT_PENDING_HUMAN_ANNOTATION`, independent validator `VALID`.
- **Mục tiêu:** Chọn thật 1.200 unique review theo corrected plan v1.1, tạo
  hai public assignment A/B blind và reserve toàn bộ selected leakage
  component trước khi con người bắt đầu.
- **Đầu vào:** Plan `q1-human-gold-plan-f8ba30c091efa284`; frame 15.553;
  năm crawled pseudo-label package chỉ dùng private sampling; curation group
  graph; prior calibration/reference reserve; Guideline V2 SHA-256
  `58ae9f19d1921fa6ce9933f85a1af2ed47a72601bfa8b1a91f48b991fb3aa1f9`.
- **Code/phương pháp đã thực thi:**
  `scripts/build_q1_human_gold_package.py` dùng deterministic seed v1.1,
  ordered-disjoint challenge selection, bảo vệ scarce Selenium group, core
  marginal selector, max 8 core/max 12 total mỗi group, opaque role ID/order,
  private crosswalk/selection ledger và full group reservation. Tạo
  `scripts/validate_q1_human_gold_package.py` độc lập dựng lại source union,
  semantic-major set và group graph; kiểm checksum/inventory, public
  blindness, quota, group cap, crosswalk và reservation closure.

### Kết quả chọn mẫu đã đo

- Tổng 1.200 unique sample ID và 1.200 unique review-text hash.
- `CORE_BALANCED` 800: đúng 100 cho mỗi tám category; mỗi category có ít nhất
  10 rating 1/2 và 10 rating 3/4.
- Core transport/curation: 57 Selenium-DOM (minimum 50), 421 clean-origin và
  379 quarantine-origin (mỗi phía minimum 320).
- `CHALLENGE_DIAGNOSTIC` 400 đóng đúng: 23 semantic-major, 170 `ESCALATE`,
  90 `REJECT_NON_REVIEW`, 55 mixed, 35 neutral và 27
  negative/rare-aspect.
- 136 selected leakage group; max 8 core và max 12 total record/group. Toàn
  bộ 4.666 record thuộc các component này được ghi vào reservation ledger.
- Prior calibration/reserved-group overlap = 0; old record = 0; confirmed
  duplicate = 0.

### Blindness và output

- A/B có cùng exact set 1.200 review nhưng thứ tự khác; opaque annotation ID
  overlap = 0. Public assignment chỉ có opaque ID, review text và text hash;
  rating/category/product/source/pseudo-label chỉ ở private ledger.
- Package:
  `data/annotations/q1_human_gold_1200_v1_20260728/`, package ID
  `q1-human-gold-package-7aa7c50d4654dd64`, reference ID
  `q1-human-gold-1200-dbc5d7e8fab66bd2`.
- Manifest SHA-256
  `a63c15aa4493dd4d173e3bfb7e97a9ab5c05bd5f664f5fcf094242ca404d98a8`;
  checksum-ledger
  `e17f2f3e6466ec0036c31bca1379c4372545502e1940ac63e3d32e88cf43fe3b`;
  selection report
  `f87bb0747dcc54b43c9668ecc3e45776da7503b5da3a3a4cfd1337ceaa033de5`.
- **Quyết định:** Package đã khóa nhưng trạng thái vẫn
  `PENDING_HUMAN_ANNOTATION`; không gọi nhãn AI là gold và chưa chia
  dev/test.
- **Giới hạn:** Deterministic balanced sample không đại diện prevalence;
  4.666 reserved record làm giảm future training pool. Hai human FINAL, IAA,
  adjudication và gold split chưa tồn tại.
- **Next dependency:** Tích hợp package vào một workbench với explicit blind
  mode, mode-bound persistence/export, generic start script và browser smoke
  test trước khi bàn giao A/B.

## TASK-20260728-038 — Tích hợp và validate một Q1 annotation workbench ba mode

- **Trạng thái:** ĐÃ THỰC THI; validation release `VALID`, independent
  validator `VALID`.
- **Mục tiêu:** Đưa Q1 package 1.200 vào UI an toàn, đồng thời tích hợp blind
  A/B, AI-assisted review và expert adjudication trong một workbench nhưng
  ngăn draft/export/input bị dùng chéo mode.
- **Đầu vào:** Package
  `q1-human-gold-package-7aa7c50d4654dd64`; workbench v1; Guideline V2;
  legacy blind/AI-review draft dùng để backward-compatibility test; synthetic
  one-conflict A/B fixture chỉ dùng kiểm UI adjudication.

### Code/phương pháp đã thực thi

- Bump workbench/export/workspace lên UI `2.0.0`, export `2.0.0`, workspace
  `2.0.0`; assignment schema giữ `1.0.0` để không sửa frozen public package.
- `serve.py` bắt buộc `--mode blind|ai-review|adjudication`; startup
  fail-closed nếu role/input không khớp. Server phát checksum-bound
  `/workflow.json`; blind không có suggestion/adjudication endpoint.
- IndexedDB v2 dùng session key
  `assignment_id + workflow_payload_sha256`; export nhúng exact workflow
  envelope. Import draft phải khớp assignment, role, workflow và guideline.
- Role được mở rộng A/B/REVIEWER/ADJUDICATOR. Adjudication input có strict
  schema/checksum, A/B semantic validation, disagreement list, source-apply
  event và `ADJUDICATION_DECISION`; manual override bắt buộc ghi lý do mới.
- UI bỏ copy hard-code 200, render theo `item_count`, có mode banner rõ, A/B
  comparator chỉ ở adjudication, public blind view không hiển thị sampling
  metadata. `start.ps1` mặc định Q1 1.200; thêm
  `start_workbench.ps1` tổng quát; README được cập nhật.
- Python export validator chấp nhận v2 mode-bound export và vẫn kiểm legacy
  v1. Blind v2 cấm assisted/adjudication event; AI-review bắt buộc matching
  suggestion-seed event; adjudication FINAL bắt buộc decision event.

### Test và kết quả đo

- 13/13 test command pass: Python compile, JS syntax, 8 semantic unit test,
  3 workflow-contract test, 3 server-contract test, blind/AI-review/
  adjudication browser fixture, browser thật trên Q1 1.200, package/plan
  independent validator và hai legacy-draft compatibility check.
- Q1 browser: tiến độ 1/1.200 được phục hồi sau reload; mode
  `BLINDED_INDEPENDENT_ANNOTATION`; source metadata scan không phát hiện
  rating/category/product/transport/pseudo-status trong public response;
  external resource host chỉ `127.0.0.1`; mobile horizontal overflow = 0.
- AI-review legacy package phục hồi 1/200; adjudication fixture hiển thị
  comparator và ghi decision `ACCEPT_A`.
- Legacy v1 drafts vẫn `VALID_DRAFT_STRUCTURE`: blind 10/200 completed và
  AI-review 88/200 completed; được báo `LEGACY_UNBOUND_WORKFLOW`, không giả
  là v2-bound.

### Validation release, quyết định và giới hạn

- Release:
  `docs/audits/q1_annotation_workbench_validation_v1_20260728/`, validation
  ID `q1-workbench-validation-523436373158d2ee`, 13 test log, 3 screenshot và
  frozen source/test provenance.
- Manifest SHA-256
  `bbcca339a56e63479b1458dad2ed2d40341767cb7f3fbfbd8750cfd20f1a0618`;
  checksum-ledger
  `d14987b76baf0fe408fd2366de842a698afc9a10d06a1fc707784798356e0f73`;
  report
  `621f250d45a2f30440813061ec8f81b040174c667a71fd1f8fe707a1de862359`.
- **Quyết định:** Một workbench/ba explicit mode; không tạo hai hệ thống.
  Dense research-workstation layout được giữ, chỉ targeted-evolution cho
  mode identity/comparator/persistence; không thay đổi annotation semantics.
- **Giới hạn:** Adjudication test là fixture, không phải Q1 adjudication
  thật. Browser plumbing PASS không chứng minh label accuracy. Chưa có human
  A/B FINAL, IAA, expert outcome hoặc dev/test release.
- **Next dependency:** Hai người thật chạy blind assignment A/B; mỗi bên
  export FINAL và independent-validate/freeze. Sau đó mới tính IAA và build
  real adjudication package.

## TASK-20260728-039 — Q1 human-annotation handoff và runbook

- **Trạng thái:** ĐÃ THỰC THI phần bàn giao/preflight; human annotation vẫn
  `NOT_STARTED`.
- **Mục tiêu:** Biến output kỹ thuật thành quy trình vận hành rõ cho hai
  annotator, ngăn submission làm hỏng frozen package và định thứ tự hậu xử lý
  phù hợp paper Q1.
- **Đầu vào:** Package/plan/workbench validation đều `VALID`; assignment A/B
  1.200; closed-inventory contract; UI autosave/export v2.
- **Phương pháp đã thực thi:** Tạo
  `docs/Q1_HUMAN_ANNOTATION_RUNBOOK.md`; ghi vai trò A/B, preflight command,
  launch command, per-review checklist, backup cadence, FINAL validation
  command và post-FINAL stage gates. PowerShell parser xác nhận syntax ba
  launcher `PASS`.
- **Quyết định:** Cần hai người thật khác nhau; A/B không trao đổi draft/nhãn
  trước freeze. Draft/FINAL/validated output phải nằm **ngoài**
  `q1_human_gold_1200_v1_20260728/` vì thêm file vào package sẽ phá closed
  checksum inventory. Suggested validated outputs là hai versioned sibling
  release `q1_human_gold_a_validated_v1_20260728` và
  `q1_human_gold_b_validated_v1_20260728`.
- **Acceptance gate:** Mỗi FINAL phải có 1.200 valid completed record,
  `workflow_mode=BLINDED_INDEPENDENT_ANNOTATION` và Python validator
  `VALID_FINAL`. IAA chỉ chạy khi cả hai gate pass.
- **Output:** Runbook trên và README workbench đã được sửa đường dẫn output
  để không ghi vào frozen package.
- **Kết quả đo:** Human annotation/IAA/adjudication/dev-test count được tạo
  trong task này đều bằng 0; đây là bàn giao thật, không dùng smoke fixture
  làm nhãn nghiên cứu.
- **Giới hạn:** Thời gian/chất lượng gán phụ thuộc hai annotator và expert
  bên ngoài code. Nếu chỉ một người làm thì chỉ được gọi single-annotator
  reference, không được báo double-blind gold/IAA.
- **Next dependency:** Annotator A chạy port 8765, annotator B chạy port 8766
  trên browser profile/máy riêng; gửi lại hai FINAL sau khi đủ 1.200/1.200 để
  validate, tính IAA và tạo adjudication package thật.

## TASK-20260728-040 — Đánh giá Memory Caching cho base ABSA

- **Trạng thái:** ĐÃ THỰC THI phần đọc paper, audit code và đo dữ liệu; **chưa
  sửa model, chưa train và chưa có kết quả accuracy/latency ABSA**.
- **Mục tiêu:** Đọc paper `2602.24281v1`, xác định mức tương thích với base
  multi-polarity ABSA hiện tại, ngăn việc áp dụng một kỹ thuật long-context
  không phù hợp phân phối input, và định nghĩa thí nghiệm/ablation có thể kiểm
  chứng cho hướng paper Q1.
- **Đầu vào:** Local PDF
  `C:\Users\Luc\OneDrive\Documents\research paper\2602.24281v1.pdf`; bản HTML
  chính thức arXiv; model code trong `legacy/system/`; sáu canonical
  pseudo-label release gồm new clean, quarantine, delta và legacy-old. Không
  dùng/sửa `data/raw/`.
- **Phương pháp đã thực thi:** Đọc abstract, method, bốn MC variants, phân tích
  complexity/segmentation, ba backbone proof-of-concept, toàn bộ bảng thí
  nghiệm/ablation/efficiency, conclusion và experimental appendix. Đối chiếu
  forward/loss/input contract của PhoBERT, XLM-R, BiLSTM và CNN-BiLSTM. Dùng
  local `vinai/phobert-base` tokenizer, có special tokens và không truncate,
  đo chiều dài 35.902 canonical pseudo-labeled review; kiểm product sequence
  inventory của base/delta corpus.
- **Kết quả đo:** Tổng token p50/p90/p95/p99/max lần lượt
  `33/56/69/110/600`; 17.259 record ≤32, 16.402 record 33–64, 2.001 record
  65–128, 200 record 129–256 và chỉ 40/35.902 (0,111%) >256; một record >512.
  Inventory gồm 32.401 `LABELED`, 2.613 `ESCALATE`, 888
  `REJECT_NON_REVIEW`; 35.014 `LABELED/ESCALATE` có 2.317 review chứa ít nhất
  một pseudo mixed label. Các số label này chỉ dùng thiết kế, không được gọi là
  human prevalence/gold.
- **Đánh giá code:** Base Transformer là PhoBERT/XLM-R + CLS pooling + mention
  head `[N,9]` + multi-hot polarity head `[N,9,3]`; `max_length=256`. Nhánh
  BiLSTM hai tầng/hai chiều gần recurrence hơn nhưng chỉ dùng final hidden.
  Active `src/` chưa có model training; không tìm thấy active checkpoint trong
  tree hiện hành. MC không phải module có thể post-hoc gắn vào bidirectional
  PhoBERT.
- **Quyết định:** Có thể thử một `MC-ABSA Lite` paper-inspired: giữ PhoBERT
  encoder, recurrently compress hidden-state segments, cache trạng thái cuối,
  aspect-specific context gate và GRM retrieval, rồi giữ hai head
  mention/multi-polarity. Đây chỉ là nhánh ablation. Không thay base ngay,
  không pretrain Titans/DLA+MC từ đầu, không concat review để giả long context,
  và không dùng lịch sử cùng product trong primary task vì làm đổi input/evidence
  contract và tăng nguy cơ leakage.
- **Thí nghiệm đã lên kế hoạch nhưng CHƯA thực thi:** PhoBERT hiện tại;
  PhoBERT + segment attention không cache; MC-NoCache; MC-Residual; MC-GRM;
  sau đó mới MC-SSC. Ablation segment 8/16/32/all, aspect-specific/shared query,
  independent/checkpoint compressor, frozen/fine-tuned encoder và evidence
  auxiliary loss. Báo 5 seed, paired group bootstrap CI, end-to-end
  aspect–polarity F1, mixed metrics, length/multi-aspect strata, VRAM,
  throughput và p50/p95 latency.
- **Output:** Báo cáo chi tiết
  `docs/audits/PAPER_2602_24281_INTEGRATION_REVIEW_20260728.md`.
- **Giới hạn:** Paper đánh giá context 4K–32K và model 760M/1,3B trên
  30B/100B token, không đánh giá Vietnamese ABSA; arXiv v1 không liên kết
  official code repository tại thời điểm kiểm tra và một số ô ablation HTML là
  `00.0`. Dữ liệu hiện tại rất ngắn nên overhead có thể lớn hơn gain. Chưa có
  human-gold/IAA/adjudicated test, vì vậy chưa thể phát biểu MC cải thiện hệ
  thống.
- **Next dependency:** Hoàn tất hai human FINAL 1.200, independent validation,
  IAA và expert adjudication; khóa human-gold dev/test và group-aware split.
  Sau đó mới implement smoke overfit và thí nghiệm MC-ABSA Lite theo ma trận đã
  định nghĩa.

## TASK-20260729-041 — Loại Memory Caching khỏi phạm vi bài ABSA

- **Trạng thái:** ĐÃ QUYẾT ĐỊNH; không có model/data execution.
- **Mục tiêu:** Khóa quyết định không tiếp tục hướng Memory Caching sau đánh
  giá feasibility tại Task 040, tránh MC còn xuất hiện như thí nghiệm dự kiến
  trong kế hoạch mô hình của paper.
- **Đầu vào:** Kết quả Task 040 và quyết định trực tiếp của chủ dự án ngày
  2026-07-29: “không dùng MC cho bài này nữa”.
- **Thao tác đã thực thi:** Giữ nguyên Task 040 và báo cáo kỹ thuật làm audit
  trail; thêm trạng thái superseding vào
  `docs/audits/PAPER_2602_24281_INTEGRATION_REVIEW_20260728.md`. Không xóa
  lịch sử đánh giá, không sửa model, không tạo split, không train, không thay
  đổi bất kỳ release hay `data/raw/` nào.
- **Quyết định:** Loại `MC-ABSA Lite`, MC-NoCache, MC-Residual, MC-GRM,
  MC-SSC và các ablation liên quan khỏi experimental scope chính thức. Không
  báo Memory Caching như contribution, baseline hoặc planned experiment của
  bài này.
- **Kết quả đo:** Số model implementation, training run, checkpoint và data
  record bị thay đổi trong task này đều bằng 0.
- **Giới hạn:** Task này chỉ khóa phạm vi; chưa lựa chọn kiến trúc thay thế.
- **Next dependency:** Hoàn tất human A/B annotation, IAA, adjudication và
  human-gold split trước; sau đó mới chọn hướng model dựa trên đúng bài toán
  ABSA multi-polarity và phân phối review ngắn.

## TASK-20260729-042 — Xây lại active ABSA model và model-ready pseudo release

- **Trạng thái:** ĐÃ THỰC THI phần code, model-ready release, validation và
  GPU smoke test; **CHƯA chạy full-corpus training và CHƯA có kết quả
  human-gold để công bố**.
- **Mục tiêu:** Chuyển repository từ trạng thái chỉ còn collector/legacy
  model sang một active, reproducible Vietnamese multi-polarity ABSA system;
  coi dữ liệu đã đủ để phát triển mô hình theo chỉ thị của chủ dự án nhưng
  không đánh đồng pseudo label với final Q1 gold benchmark.
- **Phạm vi đã khóa:** Không dùng Memory Caching hay bất kỳ biến thể MC nào.
  Bài toán active là phát hiện 9 aspect và dự đoán tập polarity độc lập
  `negative/positive/neutral`; một aspect có thể đồng thời positive và
  negative. Neutral không được đồng tồn tại với polar label ở output cuối.

### Đầu vào và hợp đồng dữ liệu đã dùng

- Sáu canonical pseudo-label source gồm tranche 5.000, remainder 8.976,
  clean delta 613, quarantine 11.166, quarantine delta 375 và legacy-old
  9.772; tổng inventory nguồn 35.902 record.
- Chỉ status `LABELED` được dùng cho model-ready release. `ESCALATE`,
  `REJECT_NON_REVIEW`, exact duplicate và toàn bộ sample/text thuộc hai human
  reservation ledger được loại bằng ledger có lý do; không sửa/xóa
  `data/raw/` hoặc canonical annotation release.
- Nhãn được ánh xạ thành mention vector `[9]` và sentiment multi-hot
  `[9,3]`: absent=`2`, negative=`-1`, neutral=`0`, positive=`1`,
  mixed=`{negative, positive}`. Mỗi evidence span giữ exact Unicode
  `[start,end)`, aspect index và polarity index.
- Leakage group được đóng theo connected component của product, exact text,
  duplicate/near-duplicate, representative và template relation. Crawled và
  legacy source family được split độc lập rồi mới merge để tránh tình trạng
  một domain gần như chỉ nằm ở train.

### Code và phương pháp ĐÃ thực thi

- Tạo active package `src/absa_system/` gồm strict schema, release builder,
  tokenizer/evidence alignment, immutable dataset loader, model, loss,
  metrics, training, checkpoint-bound inference và CLI.
- `schema.py` fail-closed khi taxonomy, text hash, vector shape, neutral
  exclusivity hoặc evidence offset sai. `data.py` tạo versioned release,
  per-record decision ledger, manifest và `SHA256SUMS`; validator kiểm tra
  closed inventory, schema, checksum, group isolation và human reservation.
- `tokenization.py` dùng fast offset tokenizer tương thích PhoBERT để ánh xạ
  character evidence sang token mask; token ID đã được đối chiếu với
  PhoBERT tokenizer gốc trong bước smoke.
- `model.py` triển khai `AspectEvidenceModel`: PhoBERT encoder, 9 learned
  aspect query, mention-specific token attention và 27
  aspect–polarity-specific token attention. Positive và negative của cùng
  aspect có query/evidence path riêng; context được fuse với global state
  qua concat, tích và absolute difference. Checkpoint smoke chứa
  142.091.522 tham số.
- `losses.py` dùng focal BCE có per-class positive weight; sentiment loss chỉ
  tính trên aspect được mention; thêm exact-evidence attention loss,
  mention–sentiment consistency và neutral-exclusivity penalty.
- `metrics.py` báo mention macro/micro, end-to-end 27-label macro/micro,
  exact-set match, sample Jaccard, Hamming loss và mixed-label P/R/F1.
  Threshold riêng theo aspect/polarity chỉ được tune trên dev; test chỉ được
  đánh giá một lần sau khi chọn best epoch theo dev end-to-end macro-F1.
- `training.py` triển khai deterministic seed, backbone/head learning rate
  riêng, warmup-linear decay, gradient accumulation/checkpointing, AMP,
  clipping, early stopping và atomic checkpoint. Mỗi completed run được seal
  bằng manifest/checksum; validator fail-closed nếu một byte artifact đổi,
  threshold không ghi `selected_on=dev`, hoặc checkpoint/data release ID lệch.
- `inference.py` nạp model config, taxonomy và threshold trực tiếp từ
  checkpoint. CLI active: `prepare`, `validate-data`, `train`,
  `validate-run`, `predict`.
- Cấu hình frozen cho bước kế tiếp nằm ở `configs/model_data_v1.json` và
  `configs/training_v1.json`. README gốc và
  `data/model_ready/README.md` đã ghi current/superseded release cùng lệnh
  chạy.

### Model-ready release đã tạo và kết quả đo

- **Current:** `data/model_ready/absa_pseudo_v1_2_20260729/`;
  release ID `absa-model-ready-5e6d8c9306664b405624`; status
  `DEVELOPMENT_PSEUDO_MODEL_READY_NOT_GOLD`; manifest SHA-256
  `05957d1590cb4494bedcaa82f29478261e0c198ec112dc0d2ae46852387841dc`.
- Source 35.902; candidate trước exact dedup 28.267; model-ready unique
  28.266. Split: train 22.508, dev 2.861, test 2.897; leakage group tương
  ứng 7.383/876/877.
- Decision ledger: 2.613 `EXCLUDE_ANNOTATION_STATUS`, 888
  `EXCLUDE_NON_REVIEW`, 4.134 `EXCLUDE_HUMAN_GOLD_RESERVATION`, một
  `EXCLUDE_EXACT_TEXT_DUPLICATE`; còn lại 28.266 include.
- Train/dev/test group overlap đều 0; reserved sample overlap 0; reserved
  text overlap 0; sample ID và review-text hash đều unique.
- Legacy-old được phân bổ 6.877/837/836 qua train/dev/test thay vì dồn gần
  hết vào train. Mixed aspect instance là 1.173/304/351.
- Hai release thử trước được giữ nguyên để audit nhưng đã supersede:
  `absa_pseudo_v1_20260729` bị domain imbalance nghiêm trọng;
  `absa_pseudo_v1_1_20260729` đã cải thiện nhưng legacy dev/test vẫn thiếu.
  Không release nào bị ghi đè hoặc xóa.

### Validation và smoke execution đã đo

- Targeted model suite: compileall PASS và 8/8 unit test PASS, gồm strict
  schema, mixed preservation, evidence offset/token alignment, threshold
  constraints, deterministic disjoint group split, forward/loss/backward,
  closed release builder và post-seal tamper detection.
- Full historical unit discovery chạy 210 test: 204 PASS, 6 ERROR. Hai lỗi do
  system Python không cài Selenium cho collector mock; bốn lỗi còn lại là
  legacy human-UI test fixture v1 chưa truyền workflow envelope/mode bắt buộc
  của workbench v2. Các lỗi này không nằm trong `absa_system`, nhưng vẫn là
  outstanding repository-regression debt và không được ghi là toàn bộ suite
  PASS.
- GPU smoke run **đã thực thi** tại
  `artifacts/models/absa_arch_smoke_v2_20260729/` với NVIDIA RTX 3050 Laptop
  GPU, 2 train/2 dev/2 test record, một epoch, max length 64, batch 1.
  Training path gồm evidence loss, backward, checkpoint, dev threshold và
  one-time test evaluation đã hoàn tất trong 1,812 giây.
- Smoke checkpoint 568.441.023 byte, SHA-256
  `0c5b8add09e9e8defd260e68fc8189a832bc8ece1737aa170c7938fb7feacaf6`;
  artifact manifest SHA-256
  `071fe371b4a4f8cb8bc81676d7fae25af37525b8414e1f5e50e3bbc8dcf8418e`;
  `validate-run` trả `VALID`, status `SEALED_SMOKE_RUN`, 8 closed files.
- Smoke dev macro-F1 `0,148148` và test macro-F1 `0,024691` chỉ đến từ hai
  record mỗi split nên **không phải accuracy result, không dùng trong paper
  và không dùng deploy**. Checkpoint-bound prediction chỉ xác nhận inference
  schema hoạt động.

### Quyết định, giới hạn và phần CHƯA thực thi

- **Quyết định:** Dùng v1.2 làm development pseudo train/dev/test release;
  không dùng v1/v1.1. Giữ kiến trúc aspect-conditioned evidence-aware làm
  active model đầu tiên, thay cho legacy CLS-only code. Không claim đây là
  contribution Q1 duy nhất trước khi có baseline/ablation/human-gold result.
- **Giới hạn:** Toàn bộ 28.266 nhãn vẫn là LLM pseudo label, không phải gold.
  Q1 A/B annotation, IAA, expert adjudication và locked human test chưa hoàn
  tất. Do đó pseudo dev/test chỉ phục vụ engineering/model selection; final
  paper metric bắt buộc chạy trên human-adjudicated holdout không tham gia
  train/tune.
- **CHƯA thực thi:** Full 28.266-record training; multi-seed experiment;
  baseline PhoBERT CLS-only; ablation evidence/consistency/query; confidence
  interval; error analysis; calibration; latency/throughput benchmark và
  production API migration. Cấu hình hyperparameter hiện tại là frozen
  starting configuration, chưa được chứng minh tối ưu.
- **Next dependency:** Chạy một full development training bằng
  `configs/training_v1.json`, kiểm tra learning curve/rare-label behavior và
  seal artifact; sau đó thiết lập baseline/ablation cùng split/seed. Song
  song, hoàn tất human A/B → IAA → adjudication → locked human-gold test để
  có kết quả được phép báo trong paper.

## TASK-20260729-043 — Đánh giá mức độ đủ mạnh của active model cho paper Q1

- **Trạng thái:** ĐÃ THỰC THI phần audit/đối chiếu literature và quyết định
  research scope; **không sửa code, không train, không tạo kết quả mới**.
- **Mục tiêu:** Trả lời liệu `AspectEvidenceModel` hiện tại có quá đơn giản
  cho paper Q1 hay không, đồng thời tránh hai sai lầm: đánh đồng độ phức tạp
  kiến trúc với chất lượng paper, hoặc gọi một tổ hợp PhoBERT + attention
  thông thường là đóng góp mới khi chưa có novelty/experiment evidence.
- **Đầu vào đã audit:** Code và kết quả Task 042; active architecture
  PhoBERT + aspect/polarity queries + soft evidence attention + independent
  BCE/constrained post-processing; model-ready pseudo release 28.266;
  trạng thái human-gold/experiment hiện tại.
- **Literature đã đối chiếu:** Bai et al., Findings EMNLP 2024 về compound
  ABSA (`https://aclanthology.org/2024.findings-emnlp.460/`); Cabello và
  Akujuobi, Findings ACL 2024 về cải tiến ABSA đơn giản nhưng có kiểm chứng
  mạnh (`https://aclanthology.org/2024.findings-acl.394/`); Zhang et al.,
  Findings NAACL 2024 về đánh giá LLM trên 13 sentiment task/26 dataset
  (`https://aclanthology.org/2024.findings-naacl.246/`); ViCloABSA 2024 với
  7.000 Vietnamese human-annotated review, strong baselines và error analysis
  (`https://aclanthology.org/2024.paclic-1.22/`); M-ABSA 2025 với 21 ngôn
  ngữ, 7 domain, human-reviewed data và extensive baselines
  (`https://arxiv.org/abs/2502.11824`).

### Kết luận audit

- **Về engineering:** Model hiện tại không phải toy model. Nó có
  aspect-conditioned/polarity-conditioned retrieval, mixed-label contract,
  evidence auxiliary loss, class imbalance handling, dev-only threshold,
  group-aware data isolation và sealed run provenance. Nó đủ làm strong
  in-house model và đủ làm proposed-method v0/baseline nâng cao.
- **Về novelty kiến trúc:** Nếu paper chỉ claim “PhoBERT + learned aspect
  query + attention + BCE”, đóng góp quá gần các mẫu multi-head/attention
  classifier quen thuộc. Soft attention hiện chưa phải explicit rationale
  extraction; post-hoc neutral constraint và independent sigmoid chưa tạo
  một structured multi-polarity decoder mới. Ở trạng thái hiện tại, model
  đơn lẻ **chưa đủ mạnh làm contribution trung tâm của Q1 model paper**.
- **Về toàn paper:** Mô hình đơn giản không tự động loại một paper tốt; các
  công trình gần đây cho thấy một thay đổi đơn giản vẫn có giá trị khi có câu
  hỏi nghiên cứu rõ, nhiều benchmark/baseline, ablation và gain được kiểm
  chứng. Ngược lại, thêm module phức tạp nhưng thiếu human gold, baseline và
  significance không giải quyết được acceptance risk.
- **Blocker lớn hơn độ đơn giản:** Chưa có full training result, chưa có
  human-adjudicated test, IAA, baseline, ablation, multi-seed confidence
  interval hoặc error analysis. Vì vậy hiện chưa tồn tại bằng chứng thực
  nghiệm để kết luận model tốt, chứ không chỉ thiếu độ phức tạp.

### Định vị paper được chọn

- Không định vị như “một attention architecture mới” thuần túy. Hướng có tính
  nhất quán cao hơn là **Evidence-grounded Multi-Polarity ABSA under
  Human-Calibrated Pseudo-Label Supervision for Vietnamese E-commerce**.
- Ba trục đóng góp dự kiến phải gắn với nhau:
  1. corpus/data-construction protocol có provenance, dedup/leakage group,
     human reservation, double-blind annotation và audit closure;
  2. explicit evidence-grounded structured multi-polarity model;
  3. đánh giá noise/transfer/rare/mixed behavior trên locked human gold.
- Active Task-042 model được giữ làm `EAMP-v0`, không bỏ đi. Nó là mốc để đo
  xem mỗi thành phần mới có cải thiện thật hay không.

### Phần nâng cấp ĐƯỢC ĐỀ XUẤT nhưng CHƯA thực thi

- Thay soft-attention-only evidence bằng **explicit aspect–polarity evidence
  span decoder** (BIO hoặc start/end), có token/span F1 và exact-span metric;
  dùng predicted evidence pooling/gate trực tiếp cho sentiment decision.
- Thay independent BCE + post-processing đơn thuần bằng **structured
  polarity-set objective/decoder** cho năm state hợp lệ: absent, negative,
  neutral, positive, mixed; giữ khả năng giải thích positive/negative bằng
  hai span riêng và không cho neutral xung đột.
- Sau khi có human labels, thêm **human-calibrated pseudo-label weighting**
  hoặc noise-aware teacher–student objective; không cho 28.266 pseudo record
  có độ tin cậy ngang human gold một cách mặc định.
- Baseline bắt buộc: PhoBERT CLS-only; PhoBERT aspect-query không evidence;
  Task-042 full model; XLM-R/ViSoBERT encoder baseline; một
  instruction/generative ABSA baseline phù hợp tài nguyên.
- Ablation bắt buộc: bỏ evidence supervision; bỏ structured constraint; bỏ
  pseudo weighting; shared so với aspect-specific query; pseudo-only so với
  pseudo + human calibration.
- Evaluation bắt buộc: end-to-end aspect–polarity macro/micro F1, mixed F1,
  evidence token/span F1, exact set/Jaccard, rare-aspect strata, clean so với
  quarantine/legacy domain, multi-aspect/length strata; ít nhất 3–5 seed,
  paired group bootstrap confidence interval và qualitative error analysis.

### Quyết định và next dependency

- **Quyết định:** Không thay model bằng một kiến trúc lớn tùy ý chỉ để trông
  phức tạp. Giữ Task-042 làm strong base; chỉ thêm các thành phần giải quyết
  trực tiếp ba weakness có thể đo: evidence faithfulness, valid
  multi-polarity structure và pseudo-label noise.
- **Giới hạn:** “Q1” là quartile của venue, không phải ngưỡng số layer hay số
  tham số; đánh giá này là research-readiness audit, không phải bảo đảm nhận
  bài. Novelty chính thức vẫn cần systematic literature review sát thời điểm
  viết submission.
- **Next dependency:** Trước tiên chạy baseline/full Task-042 để có mốc thật.
  Đồng thời hoàn tất human-gold pipeline. Chỉ implement explicit span decoder
  và structured five-state decoder sau khi mốc baseline đã được seal, để
  ablation không bị mất điểm đối chứng.

## TASK-20260729-044 — Audit file dư thừa trước khi xin phê duyệt xóa

- **Trạng thái:** ĐÃ THỰC THI read-only inventory; **chưa xóa, chưa di chuyển,
  chưa sửa bất kỳ candidate nào**.
- **Mục tiêu:** Xác định file/thư mục có thể dọn mà không làm mất raw data,
  canonical annotation, model-ready current release, crawler session hoặc
  Q1 provenance; công khai nội dung và ảnh hưởng để chủ dự án duyệt theo nhóm.
- **Phương pháp đã thực thi:** Đo file count/dung lượng theo top-level và
  subdirectory; liệt kê file từ 5 MiB; đọc run metadata của hai smoke model;
  kiểm tra reference trong code/docs; phân tách browser cache khỏi Cookies,
  Network, Local/IndexedDB/session; kiểm tra process Chrome/ChromeDriver.
  Tại thời điểm audit không có Chrome/ChromeDriver đang chạy.

### Nhóm đề nghị xóa sau khi được duyệt

1. `artifacts/models/absa_arch_smoke_v1_20260729/` — khoảng 542,13 MiB, sáu
   file. Nội dung là checkpoint `model.pt` 542,108 MiB cùng epochs/run/test
   metrics/threshold/config của smoke 2 train + 2 dev + 2 test, một epoch.
   Run completed nhưng chưa có manifest/SHA256 closure; đã được thay bằng
   `absa_arch_smoke_v2_20260729` có cùng mục tiêu và được seal/validate.
2. Mười hai browser cache/model directory, tổng 565,66 MiB:
   `Default/Cache` 422,90; `Default/Code Cache` 30,95;
   `Default/GPUCache` 1,57; `Default/DawnGraphiteCache` 0,53;
   `Default/DawnWebGPUCache` 0,53; `GrShaderCache` 5,36;
   `ShaderCache` 0,53; `GPUPersistentCache` 0,02;
   `optimization_guide_model_store` 44,09; `component_crx_cache` 32,43;
   `WasmTtsEngine` 22,25; `OnDeviceHeadSuggestModel` 4,49 MiB dưới
   `browser-profile/lazada-hybrid/`. Đây là cache/model Chrome có thể tải/tạo
   lại. Đề xuất không chạm `Default/Network/Cookies`, Local Storage,
   IndexedDB, Sessions, Preferences hoặc `Local State`.
3. `.tmp/` — 11,30 MiB, 223 file. Nội dung gồm pre-primary/failed diagnostic
   archive ngày 2026-07-27, stdout/stderr build/audit, hai Pandoc round-trip
   text và ba UI smoke screenshot. Ba screenshot đã có bản đóng checksum
   trong `docs/audits/q1_annotation_workbench_validation_v1_20260728/`;
   kết quả cần báo đã nằm trong protocol/final release.
4. Bảy nhóm `__pycache__/` dưới `human_annotation_ui`, `legacy/system`,
   `scripts`, `src` và `tests` — khoảng 2,43 MiB, 93 bytecode file; toàn bộ
   được Python tái tạo từ source.

- **Tổng dung lượng có thể thu hồi nếu duyệt cả bốn nhóm:** khoảng
  1.121,52 MiB, tương đương 1,095 GiB.

### Nhóm chưa đề nghị xóa

- `artifacts/models/absa_arch_smoke_v2_20260729/` 542,13 MiB: là checkpoint
  active duy nhất đã seal/validate; chỉ nên xóa sau khi có full training
  checkpoint hợp lệ.
- `data/model_ready/absa_pseudo_v1_20260729/` và
  `absa_pseudo_v1_1_20260729/`, tổng 106,08 MiB: đã supersede nhưng là hai
  release khép kín ghi lại lý do sửa chiến lược split. Có thể xóa về mặt vận
  hành, nhưng hiện khuyến nghị giữ để audit Q1.
- `browser-profile/lazada-hybrid/Default/Service Worker` 74,70 MiB và toàn bộ
  cookie/session/profile database: giữ để giảm nguy cơ làm thay đổi crawler
  browser identity/session.
- `data/releases/`, `data/annotations/`, `legacy/` và `logs/`: giữ provenance,
  baseline/history; chưa có bằng chứng đủ mạnh để gọi là dư.
- Toàn bộ `data/raw/`: protected immutable input, tuyệt đối không nằm trong
  cleanup proposal.

- **Quyết định:** Không thực hiện destructive action trước khi chủ dự án duyệt
  exact group. Nếu được duyệt, phải xác minh lại absolute path nằm trong
  workspace, kiểm tra Chrome đã dừng, xóa từng group bằng literal path, đo
  dung lượng sau xóa và ghi kết quả/thứ có thể hoặc không thể phục hồi.
- **Next dependency:** Chủ dự án trả lời các nhóm được phép xóa, ví dụ
  `xóa nhóm 1, 2, 3, 4`; không suy diễn phê duyệt cho nhóm khác.

## TASK-20260729-045 — Hoãn cleanup đến khi hoàn tất toàn bộ nghiên cứu

- **Trạng thái:** ĐÃ QUYẾT ĐỊNH `DEFERRED`; **không có file/thư mục nào bị
  xóa hoặc di chuyển**.
- **Mục tiêu:** Giữ khả năng truy vết và phục hồi trong giai đoạn model,
  human-gold, baseline/ablation và paper vẫn chưa hoàn tất; tránh tiết kiệm
  dung lượng sớm nhưng làm mất artifact cần đối chiếu.
- **Đầu vào:** Cleanup audit Task 044 với bốn nhóm candidate khoảng
  1,095 GiB và quyết định trực tiếp của chủ dự án rằng nên giữ lại rồi xóa
  sau khi hoàn tất tất cả.
- **Quyết định:** Hoãn toàn bộ bốn nhóm cleanup. Không xóa smoke v1, browser
  cache, `.tmp/` hoặc `__pycache__` tại thời điểm này. Danh sách Task 044 chỉ
  là candidate inventory, không phải deletion authorization.
- **Cleanup gate dự kiến nhưng CHƯA thực thi:** Chỉ xét xóa khi đồng thời có:
  full training checkpoint đã seal/validate; baseline và ablation artifacts
  đã khóa; human A/B, IAA, adjudication và locked gold test hoàn tất; paper
  tables/figures tái tạo được; source/data/model quan trọng có backup hoặc
  checksum closure; crawler không cần browser cache/session candidate.
- **Giới hạn:** Quyết định giữ lại tạm thời tiếp tục tiêu thụ khoảng
  1,095 GiB có thể thu hồi. `data/raw/` vẫn là protected immutable input và
  không trở thành cleanup candidate kể cả sau khi paper hoàn tất.
- **Next dependency:** Tiếp tục full model experiment. Sau final research
  freeze, chạy lại read-only inventory vì dung lượng và dependency có thể đã
  thay đổi, rồi xin phê duyệt exact path một lần nữa trước khi xóa.

## TASK-20260729-046 — Tạo nhánh Git `final_absa` với active system snapshot

- **Trạng thái:** ĐÃ THỰC THI local branch preparation, validation và commit;
  **không push remote, không xóa working-tree archive/data**.
- **Mục tiêu:** Tạo một nhánh repository gọn cho active ABSA system, chỉ
  version code, cấu hình, data release dùng trực tiếp và tài liệu Markdown
  cần thiết; loại DOCX, smoke/checkpoint artifact, cookie/browser profile,
  temp/log/cache và test khỏi snapshot.
- **Đầu vào:** Working tree sau Task 045; active `src/absa_system`,
  `src/lazada_collector`, data-processing scripts, annotation workbench,
  frozen configuration và current model-ready release v1.2.
- **Phương pháp đã thực thi:** Tạo branch local `final_absa` từ
  `experiments`; stage toàn bộ tracked deletion của hệ thống cũ để chúng
  không còn xuất hiện trong branch; stage có allowlist các nhóm
  `src/`, `configs/`, `scripts/`, `human_annotation_ui/`, root launcher,
  packaging metadata, essential Markdown và duy nhất current model-ready
  v1.2. Bổ sung `.gitattributes` để JSON/JSONL và source text giữ LF, bảo vệ
  byte-level dataset checksums qua checkout.
- **File/data được thêm hoặc sửa:** 112 added, 3 modified và một rename tại
  pre-commit inventory; 168 tracked obsolete path bị xóa khỏi branch.
  Added/modified blob tổng khoảng 55,02 MiB. Bốn file lớn nhất là
  `train.jsonl` 30,36 MiB, `decision_ledger.jsonl` 14,55 MiB,
  `test.jsonl` 4,08 MiB và `dev.jsonl` 4,03 MiB; không file nào đạt giới hạn
  100 MiB.
- **Data được version:** Chỉ
  `data/model_ready/absa_pseudo_v1_2_20260729/` với 28.266 record và
  `data/model_ready/README.md`, cộng placeholder `.gitkeep` cho raw,
  releases, annotations và manifests. Raw/annotation/release local không
  được force-add vào Git.
- **Explicit exclusion:** Không stage `*.docx`, `artifacts/`, model
  checkpoint, smoke file/test, `tests/`, `.tmp/`, logs, browser profile,
  cookies, Python bytecode, hai superseded model-ready v1/v1.1, `legacy/`
  hoặc external benchmark archive. Các mục này vẫn được giữ trên filesystem
  theo quyết định deferred cleanup; không bị xóa.
- **Validation đã chạy:** Staged release SHA-256 kiểm tra trực tiếp từ Git
  index PASS 7/7; current data validator `VALID` với
  22.508/2.861/2.897 train/dev/test và 7.383/876/877 group; model unit suite
  PASS 8/8; Python compileall PASS sau khi chạy tuần tự; forbidden-path scan
  không phát hiện DOCX/smoke/artifact/cookie/test trong added/modified set.
- **Quyết định:** Nhánh này là runnable source/data snapshot, không phải nơi
  lưu binary checkpoint hoặc toàn bộ 1,46 GiB research archive. Dataset
  source provenance đầy đủ tiếp tục tồn tại local/backup và được mô tả trong
  protocol, nhưng Git chỉ chứa final model-ready engineering release.
- **Giới hạn:** `configs/model_data_v1.json` mô tả cách dựng lại v1.2 nhưng
  clone chỉ có final model-ready release; muốn rebuild từ đầu vẫn cần lấy
  canonical raw/annotation packages từ research storage riêng. Branch chưa
  được push remote trong task này.
- **Next dependency:** Push `final_absa` chỉ khi chủ dự án yêu cầu; sau clone,
  tạo environment và cài `.[browser,ml]`, validate data, rồi chạy capacity
  pilot/full training. Checkpoint sinh ra tiếp tục nằm ngoài Git.

## TASK-20260729-047 — Publish nhánh `final_absa` lên remote

- **Trạng thái:** ĐÃ THỰC THI local publication commit và push lên `origin`;
  không force-push, không sửa/xóa remote branch khác.
- **Mục tiêu:** Công bố active source/data snapshot đã kiểm tra ở Task 046
  trên remote repository để có upstream branch riêng, trong khi tiếp tục
  loại DOCX, smoke/checkpoint artifact và local research archive khỏi Git.
- **Đầu vào:** Local branch `final_absa`; snapshot commit Task 046; remote
  `origin` tại `https://github.com/Longhehehe/Real-Time-ABSA-System`.
- **Phương pháp đã thực thi:** Xác nhận current branch và tracked working tree
  sạch; cập nhật protocol Markdown; render/round-trip validate DOCX chỉ ở
  local; commit thay đổi protocol; chạy non-force
  `git push -u origin final_absa`; kiểm tra upstream và remote ref sau push.
- **Output:** Remote branch `origin/final_absa` trỏ tới cùng publication
  commit với local `final_absa`. DOCX, artifacts, raw/annotation archive,
  cookie/profile và các ignored/untracked file local không được push.
- **Kết quả/giới hạn:** Đây là source + current model-ready data publication,
  không phải model checkpoint release hoặc complete research-archive backup.
  Git history cha vẫn là lịch sử repository hiện có; task này không rewrite
  history để loại secret cũ đã từng tồn tại ở commit trước.
- **Next dependency:** Trên máy chạy model, checkout/pull `final_absa`, tạo
  environment ML, validate release v1.2 và chạy capacity pilot. Chỉ tạo tag
  hoặc GitHub Release sau khi full baseline artifact đã seal.

## TASK-20260729-048 — Viết lại bộ triển khai Ubuntu Server cho `final_absa`

- **Trạng thái:** ĐÃ THỰC THI việc viết tài liệu, tạo launcher và kiểm tra
  tĩnh/dry-run tại máy Windows; **CHƯA thực thi setup hoặc training trên một
  Ubuntu/NVIDIA server thật**.
- **Mục tiêu:** Thay thế hướng dẫn server cũ dành cho
  Docker/Airflow/Kafka/experiment runner bằng một đường triển khai gọn, đúng
  active source `src/absa_system`, current model-ready v1.2 và frozen training
  config của nhánh `final_absa`.
- **Đầu vào:** Active CLI trong `src/absa_system/cli.py`;
  `configs/training_v1.json`; immutable release
  `data/model_ready/absa_pseudo_v1_2_20260729/`; packaging extras trong
  `pyproject.toml`; tài liệu/script server cũ dưới `legacy/system/`; official
  PyTorch installation selector và Hugging Face Transformers offline-cache
  documentation.
- **Phương pháp/code đã thực thi:** Tạo
  `scripts/setup_final_absa_server.sh` với bốn action
  `setup|validate|pilot|full`; tạo
  `scripts/deploy_final_absa_server.ps1` để Windows kiểm tra SSH, upload một
  bootstrap Bash có tên tạm bằng GUID, truyền argument đã quote và xóa đúng
  file tạm sau khi chạy; tạo `docs/SERVER_SETUP_FINAL_ABSA.md`; liên kết
  hướng dẫn từ `README.md`.
- **Các gate được cài đặt:** Linux và Python >=3.11; checkout update chỉ bằng
  fast-forward và dừng nếu có tracked modification; virtual environment
  `.venv-model`; inventory NumPy/PyTorch/Transformers/CUDA; bắt buộc
  `validate-data`; tải hoặc xác minh `vinai/phobert-base`; chặn yêu cầu CUDA
  khi PyTorch không thấy GPU; chặn full CPU training nếu không có xác nhận
  explicit; không ghi đè run name; pilot cố định 1.000/200/200 record và một
  epoch; nối `validate-run` sau training; hỗ trợ detached `tmux`, offline
  Hugging Face cache và runtime memory overrides.
- **Quyết định:** Script không tự cài/sửa NVIDIA driver; PyTorch CUDA wheel
  index không hard-code mà phải lấy từ official selector phù hợp server tại
  thời điểm setup. Model server mặc định không nhận cookie, raw/private
  annotation archive hoặc local checkpoint. Dataset v1.2 lấy từ Git và phải
  pass checksum/schema/group-isolation validation trước training. Giảm
  `max_length` được xem là thay đổi experiment, không phải OOM fix âm thầm.
- **Kết quả đo đã thực thi:** PowerShell AST parser PASS; PowerShell `-Help`
  PASS; dry-run cho `setup`, low-memory detached `pilot` và detached `full`
  PASS, đồng thời xác nhận không mở SSH/thay đổi remote; Bash `bash -n` PASS
  bằng Git Bash; Bash `--help` PASS; LF check của Bash cho 0 byte CR và Git
  attribute `eol=lf`; CLI argument đối chiếu khớp với các option hiện có của
  `absa_system train`, `validate-data` và `validate-run`. `shellcheck` không
  có trên máy local nên không được tuyên bố là đã chạy.
- **Output:** Một runbook khoảng 13 phần gồm prerequisite, Windows SSH
  deployment, direct server setup, validation, capacity pilot, full
  training, offline mode, update policy, fault handling và security; hai
  launcher không chứa credential; README có quick-start link.
- **Giới hạn:** Chưa có server address/SSH key trong phạm vi task nên chưa
  chạy apt, clone remote, pip install, PhoBERT download, `nvidia-smi`,
  capacity pilot hoặc full training thật. Khuyến nghị 12 GB VRAM chỉ là điểm
  bắt đầu vận hành; minimum VRAM và throughput phải được đo bằng capacity
  pilot trên GPU đích. Training CLI hiện chưa resume optimizer/mid-epoch.
- **Planned nhưng CHƯA thực thi:** Chọn official PyTorch index theo
  GPU/driver; chạy remote `setup`; chạy `validate`; chạy detached capacity
  pilot; kiểm tra OOM, runtime, sealed artifact; sau khi pilot pass mới chạy
  full experiment.
- **Next dependency:** Có Ubuntu server và SSH destination; chạy
  `deploy_final_absa_server.ps1 -Action setup`, sau đó `-Action pilot
  -Detach`. Ghi lại GPU model, driver, wheel, wall-clock, peak VRAM và kết quả
  `validate-run` trước khi chốt full-training configuration.
