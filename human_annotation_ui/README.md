# Human ABSA Annotation Desk

Đây là **một workbench duy nhất** cho ba stage tách biệt:

1. `BLINDED_INDEPENDENT_ANNOTATION`: A/B gán độc lập, không thấy AI/source
   metadata.
2. `AI_ASSISTED_HUMAN_VERIFICATION`: reviewer thấy rõ gợi ý AI và phải xác
   nhận/sửa; output không được gọi là double-blind gold.
3. `EXPERT_ADJUDICATION`: chỉ mở sau hai FINAL A/B đã validate và IAA đã
   publish; expert so sánh A/B và giải quyết disagreement.

Workflow mode, assignment hash và auxiliary-input hash được bind vào
IndexedDB/export. Không thể khôi phục draft chéo mode. Toàn bộ công cụ nằm
ngoài `src/` và không thay đổi collector/runtime.

## Q1 human-gold package hiện hành

```text
data/annotations/q1_human_gold_1200_v1_20260728/
```

- 1.200 unique review: core 800 + challenge diagnostic 400;
- cùng source set cho A/B, opaque ID và thứ tự độc lập;
- 136 selected leakage group, toàn bộ 4.666 group-member record đã reserve;
- public assignment không có rating/category/product/source/pseudo-label;
- trạng thái hiện tại: `BUILT_PENDING_HUMAN_ANNOTATION`, chưa phải gold.

Chạy annotator A:

```powershell
.\human_annotation_ui\start.ps1 -Role A -Port 8765
```

Annotator B là người khác và chạy trên máy/browser profile riêng:

```powershell
.\human_annotation_ui\start.ps1 -Role B -Port 8766
```

Lệnh tổng quát tương đương:

```powershell
.\human_annotation_ui\start_workbench.ps1 `
  -Mode Blind -Role A -Port 8765
```

UI autosave vào IndexedDB của đúng browser profile. Đóng tab/máy rồi mở lại
đúng assignment sẽ tiếp tục tiến độ; vẫn phải tải **Sao lưu DRAFT** sau mỗi
20–25 review. Browser draft không phải artifact nghiên cứu.

Sau khi A hoặc B đủ 1.200/1.200 và xuất FINAL:

```powershell
.\.venv\Scripts\python -X utf8 -m human_annotation_ui.validate_export `
  --assignment .\data\annotations\q1_human_gold_1200_v1_20260728\assignments\annotator_a.assignment.json `
  --export <DUONG_DAN_FINAL_A.json> `
  --require-final `
  --output .\data\annotations\q1_human_gold_a_validated_v1_20260728
```

Đổi assignment/output tương ứng cho B. Không tính IAA cho tới khi cả hai
validator đều trả `VALID_FINAL`; không tạo adjudication package trước khi IAA
được publish. Không đặt submission/validated output vào frozen package vì
package dùng closed checksum inventory.

## Legacy calibration package 200

Package hiện hành:

```text
data/annotations/human_reference_v1_20260726/
```

Thiết kế gồm:

- 150 review representative, lấy mẫu tỷ lệ từ clean-core 20.622;
- 50 review challenge theo các coverage proxy đã đăng ký trước;
- cùng 200 source review cho A và B;
- thứ tự và opaque annotation ID khác nhau giữa A và B;
- public assignment chỉ có opaque ID, `reviewContent` và SHA-256 của text.

Rating, category, product/shop ID, nguồn crawler, cleaning flag, panel/bin,
nhãn cũ và nhãn LLM không tồn tại trong public assignment hoặc DOM của UI.

## Chạy legacy calibration UI

Từ PowerShell ở repository root, annotator A chạy:

```powershell
.\human_annotation_ui\start.ps1 -Role A `
  -PackageRoot data\annotations\human_reference_v1_20260726
```

Annotator B chạy trên browser profile hoặc máy riêng:

```powershell
.\human_annotation_ui\start.ps1 -Role B -Port 8766 `
  -PackageRoot data\annotations\human_reference_v1_20260726
```

Nếu PowerShell chặn script, dùng lệnh trực tiếp:

```powershell
.\.venv\Scripts\python -X utf8 -m human_annotation_ui.serve `
  --mode blind `
  --assignment .\data\annotations\human_reference_v1_20260726\assignments\annotator_a.assignment.json `
  --open
```

Server chỉ bind `127.0.0.1`, không dùng CDN, analytics hoặc API ngoài. Đóng tab
không dừng server; nhấn `Ctrl+C` trong terminal để dừng.

## Quy trình gán một review

1. Nhập mã người gán giả danh, ví dụ `ANN-A-01`. Mã bị khóa sau review hoàn
   tất đầu tiên.
2. Đọc toàn bộ review và tick xác nhận đã đọc.
3. Chọn trạng thái review.
4. Xét đủ chín aspect độc lập. Không có nhãn mặc định.
5. Với mọi nhãn mentioned (`-1`, `0`, `1`, `"1, -1"`), bôi đen exact text ở
   khung review và thêm làm evidence.
6. Thêm uncertainty code nếu chưa chắc. UI tự chuyển sang `ESCALATE`.
7. Bấm **Kiểm tra**, rồi **Hoàn tất và sang review kế**.

Nút điền các ô trống thành `2` chỉ mở sau xác nhận đã đọc và luôn ghi audit
event. Không dùng nút này để thay cho việc xét từng aspect.

UI autosave vào IndexedDB của đúng browser profile. Hãy bấm **Sao lưu** sau
mỗi khoảng 25 review và giữ file JSON ở nơi an toàn. Không gửi file A cho B
hoặc ngược lại.

## Human-check gợi ý AI

Luồng này khác với double-blind annotation. Nó chỉ dùng sau khi đã tạo gói
`human_reference_ai_preannotation_v1_20260726`; mọi nhãn/evidence ban đầu là
gợi ý AI và chưa phải human gold.

```powershell
.\human_annotation_ui\start_ai_review.ps1
```

Script mặc định dùng port `8770` và một assignment ID riêng, nên không ghi đè
phiên role A đã làm ở port `8765`. Người kiểm tra phải đọc từng review, xác
nhận hoặc sửa gợi ý, rồi bấm hoàn tất. Export cuối của luồng này chỉ được mô tả
là **AI-assisted, human-verified**; không được báo cáo như hai annotator
double-blind độc lập.

Sau khi human-check đủ 200/200 và xuất FINAL, kiểm tra bằng assignment riêng:

```powershell
.\.venv\Scripts\python -X utf8 -m human_annotation_ui.validate_export `
  --assignment .\data\annotations\human_reference_ai_preannotation_v1_20260726\human_check\ai_review.assignment.json `
  --export <DUONG_DAN_FILE_FINAL_HUMAN_CHECK.json> `
  --require-final `
  --output .\data\annotations\human_reference_ai_preannotation_v1_20260726\validated\human_check_v1
```

## Expert adjudication mode

Mode này dùng cùng workbench nhưng fail-closed nếu thiếu assignment role
`ADJUDICATOR` hoặc adjudication input đã checksum-bind với hai human export và
IAA report:

```powershell
.\human_annotation_ui\start_workbench.ps1 `
  -Mode Adjudication `
  -Assignment <ADJUDICATOR_ASSIGNMENT.json> `
  -Adjudication <A_B_COMPARISON_INPUT.json> `
  -Guideline <ABSA_ANNOTATION_GUIDELINE_V2.md> `
  -Port 8775
```

Hiện Q1 package chưa có hai human FINAL nên chưa được tạo hai file đầu vào
này. Không dùng fixture smoke-test làm annotation thật.

## Evidence và Unicode

Offset evidence là zero-based Unicode code-point, half-open `[start, end)`.
UI kiểm tra lại exact substring và occurrence. Điều này tránh lệch offset khi
review có emoji hoặc ký tự ngoài Basic Multilingual Plane.

## Xuất và kiểm tra FINAL

**Xuất FINAL** chỉ mở khi đúng 200/200 review đã hoàn tất và hợp lệ. Sau khi
khóa, UI chuyển sang chỉ đọc.

File tải xuống chưa tự động trở thành gold. Curator phải chạy validator Python:

```powershell
.\.venv\Scripts\python -X utf8 -m human_annotation_ui.validate_export `
  --assignment .\data\annotations\human_reference_v1_20260726\assignments\annotator_a.assignment.json `
  --export <DUONG_DAN_FILE_FINAL_A.json> `
  --require-final `
  --output .\data\annotations\human_reference_v1_20260726\validated\annotator_a_v1
```

Đổi assignment/output tương ứng cho B. Output validator là thư mục mới và
không ghi đè bản cũ.

IAA phải được tính trên hai bản A/B đã khóa, trước adjudication. Expert chỉ mở
adjudication sau khi cả A và B hoàn tất. Bộ 200 này không dùng làm practice.

Nếu dùng reference này để đánh giá model, phải giữ toàn bộ dòng cùng leakage
group trong `private/group_reservations.jsonl` khỏi training. Nếu chỉ dùng làm
semantic quality gate cho LLM, vẫn phải báo riêng metric representative và
challenge; không gộp 200 thành một population estimate tùy tiện.
