# Hướng dẫn gán nhãn ABSA Multi-Polarity V2

**Mã tài liệu:** `ABSA-ANNOTATION-GUIDELINE-V2`  
**Phiên bản:** `2.0.0`  
**Ngày hiệu lực:** 2026-07-25  
**Ngôn ngữ dữ liệu:** Tiếng Việt  
**Miền dữ liệu:** Đánh giá sản phẩm thương mại điện tử  
**Loại bài toán:** Aspect Category Sentiment Analysis (ACSA), đa khía cạnh và
đa cực tính ở cấp toàn bộ review

## 1. Mục đích và phạm vi

Tài liệu này định nghĩa duy nhất một chuẩn gán nhãn cho bộ dữ liệu ABSA gồm
chín khía cạnh. Nó phải được dùng thống nhất trong:

- đào tạo annotator;
- pilot annotation;
- gán nhãn độc lập;
- tính inter-annotator agreement (IAA);
- adjudication;
- kiểm tra chất lượng bản nhãn cuối;
- chuyển nhãn sang tensor phục vụ mô hình ML/DL.

Tài liệu này không hướng dẫn thu thập hoặc tự động làm sạch review. Tuy nhiên,
nếu annotator phát hiện nội dung không phải review, spam, quảng cáo hoặc văn
bản hệ thống còn sót lại, họ phải chuyển mẫu đó về hàng đợi kiểm duyệt dữ liệu,
không được biến nó thành một mẫu “không nhắc khía cạnh” bằng cách gán toàn bộ
nhãn `2`.

Các từ khóa **PHẢI**, **KHÔNG ĐƯỢC**, **NÊN** trong tài liệu có tính quy phạm.

## 2. Định nghĩa chính thức của đơn vị gán nhãn

### 2.1. Đơn vị annotation

Một đơn vị annotation là **toàn bộ nội dung của một review**, có thể gồm một
hoặc nhiều câu, nhiều vế hoặc nhiều dòng. Không tách từng câu thành các mẫu độc
lập ở bước gán nhãn.

Với mỗi review, annotator phải đưa ra đúng chín quyết định, theo đúng thứ tự:

1. Chất lượng sản phẩm
2. Hiệu năng & Trải nghiệm
3. Đúng mô tả
4. Giá cả & Khuyến mãi
5. Vận chuyển
6. Đóng gói
7. Dịch vụ & Thái độ Shop
8. Bảo hành & Đổi trả
9. Tính xác thực

Một review có thể nhắc đến nhiều khía cạnh. Các khía cạnh phải được xét độc
lập; cảm xúc ở một khía cạnh không được tự động truyền sang khía cạnh khác.

### 2.2. Đầu vào annotator được phép nhìn thấy

Trong vòng gán nhãn độc lập, annotator chỉ được thấy:

- một mã annotation ngẫu nhiên, không mang ý nghĩa;
- nguyên văn trường `reviewContent`;
- phiên bản guideline và danh mục ví dụ chuẩn;
- các ô để nhập chín nhãn, evidence và mã không chắc chắn.

Annotator phải bị làm mù đối với các trường sau:

- số sao/rating;
- tên hoặc ID sản phẩm, URL sản phẩm và danh mục sản phẩm;
- ID người mua, người bán hoặc shop;
- nguồn crawler, API/DOM transport, ngày crawl;
- cờ duplicate, spam, template hoặc quyết định của bước cleaning;
- nhãn cũ, nhãn do mô hình/LLM dự đoán;
- nhãn của annotator khác;
- phân bố nhãn hiện tại của dataset.

Mục đích là đảm bảo nhãn chỉ phản ánh bằng chứng trong chính văn bản mà mô hình
sẽ nhận ở thời điểm suy luận. Nếu một nghiên cứu sau này đưa thêm product title
hoặc metadata vào input mô hình, đó phải là một protocol và một phiên bản
annotation khác; không được âm thầm dùng metadata cho một phần annotator.

### 2.3. Không chỉnh sửa review trong lúc gán nhãn

Annotator không được sửa chính tả, dịch, rút gọn, xóa emoji hoặc viết lại nội
dung review. Nếu có thể hiểu được tiếng lóng, viết tắt hoặc lỗi gõ từ ngữ cảnh,
hãy gán nhãn theo nghĩa đó. Nếu không đủ chắc chắn, vẫn đưa ra nhãn tạm thời và
đánh dấu mã không chắc chắn phù hợp.

## 3. Schema đầu ra và ý nghĩa nhãn

### 3.1. Schema legacy 10 cột

Bản dữ liệu nhãn dùng để huấn luyện phải có đúng các cột sau, đúng tên và đúng
thứ tự:

1. `reviewContent`
2. `Chất lượng sản phẩm`
3. `Hiệu năng & Trải nghiệm`
4. `Đúng mô tả`
5. `Giá cả & Khuyến mãi`
6. `Vận chuyển`
7. `Đóng gói`
8. `Dịch vụ & Thái độ Shop`
9. `Bảo hành & Đổi trả`
10. `Tính xác thực`

Các trường provenance, evidence và lịch sử adjudication phải được lưu ở bảng
audit riêng, liên kết bằng `sample_id` và `review_text_sha256`; không chèn chúng
vào schema huấn luyện legacy.

### 3.2. Năm giá trị nhãn hợp lệ

| Giá trị trong ô | Ý nghĩa | Mention | Tensor sentiment |
|---|---|---:|---|
| `2` | Khía cạnh hoàn toàn không được nhắc đến | `0` | `[0, 0, 0]` |
| `-1` | Khía cạnh được nhắc với cảm xúc tiêu cực | `1` | `[1, 0, 0]` |
| `1` | Khía cạnh được nhắc với cảm xúc tích cực | `1` | `[0, 1, 0]` |
| `0` | Khía cạnh được nhắc nhưng không có cực tính rõ | `1` | `[0, 0, 1]` |
| `"1, -1"` | Cùng khía cạnh có cả bằng chứng tích cực và tiêu cực | `1` | `[1, 1, 0]` |

**Thứ tự tensor bắt buộc của hệ thống hiện hữu là**
`[negative, positive, neutral]`. Không được đổi thành
`[negative, neutral, positive]` hoặc một thứ tự khác nếu chưa thay toàn bộ data
loader, model, checkpoint và evaluator.

### 3.3. Blank khác hoàn toàn nhãn `2`

Đây là quy tắc quan trọng nhất về định dạng:

- **Ô trống:** mẫu chưa được annotator xử lý hoặc quy trình annotation chưa
  hoàn tất.
- **Nhãn `2`:** annotator đã đọc review và xác nhận không có bằng chứng về
  khía cạnh đó.

Bản annotation đang làm có thể chứa ô trống. Bản gold/final dùng cho training,
development hoặc test **không được có bất kỳ ô nhãn trống nào**. Không được
chuyển hàng loạt ô trống thành `2` để làm file “đủ nhãn”.

Data loader cuối phải chạy ở chế độ strict và báo lỗi khi gặp blank, nhãn lạ
hoặc cột thiếu. Không được âm thầm đổi một giá trị không parse được thành
neutral hoặc absent.

### 3.4. Biểu diễn canonical

Giá trị mixed trong file cuối phải là đúng chuỗi `"1, -1"`. Các biến thể sau
không hợp lệ trong canonical output:

- `1,-1`
- `-1, 1`
- `[1, -1]`
- `mixed`
- `conflict`

Importer có thể nhận diện một số biến thể để sửa trong bước validation, nhưng
phải ghi log; exporter chỉ được sinh biểu diễn canonical.

Taxonomy V2 không hỗ trợ các tổ hợp `1, 0`, `-1, 0` hoặc `1, -1, 0`. Khi một
khía cạnh có cả mệnh đề trần thuật và mệnh đề mang cảm xúc rõ, nhãn mang cảm xúc
chiếm ưu thế. Chỉ cặp positive và negative tạo thành mixed.

## 4. Định nghĩa và ranh giới của chín khía cạnh

### 4.1. Chất lượng sản phẩm

**Bao gồm:**

- chất liệu, độ bền và độ chắc chắn;
- độ hoàn thiện, đường may, keo dán, bề mặt, lỗi ngoại quan;
- thiết kế, kiểu dáng, màu sắc và kích thước vật lý;
- cảm nhận chung có đích rõ là “hàng”, “sản phẩm” hoặc một bộ phận vật lý;
- hư hỏng hoặc khiếm khuyết của bản thân sản phẩm.

**Không bao gồm:**

- tốc độ, pin, khả năng hoạt động hoặc trải nghiệm sử dụng thực tế: dùng
  `Hiệu năng & Trải nghiệm`;
- việc màu/size/mẫu khác listing: dùng `Đúng mô tả`;
- hộp vận chuyển móp, cách bọc hoặc chống sốc: dùng `Đóng gói`;
- real/fake/chính hãng: dùng `Tính xác thực`.

**Ví dụ:**

- “Vải dày dặn, đường may đẹp.” → `1`
- “Sản phẩm bị trầy và ọp ẹp.” → `-1`
- “Áo màu xanh, size M.” → `0`
- “Hàng đẹp nhưng có nhiều chỉ thừa.” → `"1, -1"`
- “Mình đặt màu xanh nhưng nhận màu đỏ.” → khía cạnh này `2`; `Đúng mô tả=-1`

### 4.2. Hiệu năng & Trải nghiệm

**Bao gồm:**

- sản phẩm hoạt động đúng hay lỗi;
- tốc độ, độ mượt, hiệu quả, pin, âm thanh, nhiệt độ khi sử dụng;
- tính tiện dụng, độ thoải mái và trải nghiệm sử dụng thực tế;
- tác dụng hoặc kết quả sau khi dùng sản phẩm.

**Không bao gồm:**

- vật liệu, ngoại hình hoặc lỗi hoàn thiện thuần vật lý;
- thông số chỉ được đối chiếu với listing, nếu trọng tâm là “khác mô tả”;
- mong đợi hiệu năng chỉ được suy ra từ rating hoặc tên sản phẩm.

**Ví dụ:**

- “Máy chạy mượt, pin dùng cả ngày.” → `1`
- “Cắm lên không nhận, dùng rất nóng.” → `-1`
- “Pin dung lượng 5.000 mAh.” → `0`
- “Máy chạy nhanh nhưng thỉnh thoảng bị treo.” → `"1, -1"`
- “Chưa dùng nên chưa biết có tốt không.” → `0`

### 4.3. Đúng mô tả

**Bao gồm:**

- sự phù hợp giữa hàng nhận được với hình ảnh, mô tả, thông số hoặc quảng cáo;
- đúng/sai màu, size, mẫu, phiên bản, số lượng hoặc phụ kiện;
- giao nhầm sản phẩm so với đơn đặt.

Khía cạnh này cần có quan hệ đối chiếu rõ hoặc được diễn đạt bằng các từ như
“đúng mô tả”, “giống hình”, “khác hình”, “đặt X giao Y”. Chỉ nói một thuộc tính
của sản phẩm không tự động tạo nhãn `Đúng mô tả`.

**Ví dụ:**

- “Hàng y hình, đúng màu mình đặt.” → `1`
- “Quảng cáo cotton nhưng nhận về là vải pha.” → `-1`
- “Shop ghi kích thước 20 cm.” → `0` nếu chỉ thuật lại listing
- “Màu đẹp.” → khía cạnh này `2`; có thể là `Chất lượng sản phẩm=1`
- “Khác hình một chút nhưng mẫu thực tế vẫn rất đẹp.” →
  `Đúng mô tả=-1`, `Chất lượng sản phẩm=1`

### 4.4. Giá cả & Khuyến mãi

**Bao gồm:**

- đắt, rẻ, hợp lý, đáng tiền hoặc không xứng giá;
- mã giảm giá, voucher, flash sale, quà khuyến mãi;
- so sánh giá và lợi ích nhận được.

**Không bao gồm:**

- phí vận chuyển, trừ khi review đang nhận xét tổng chi phí và tách rõ phần giá;
- “nhận xu để viết review” nếu đó chỉ là boilerplate của nền tảng; trường hợp
  này phải chuyển cleaning review.

**Ví dụ:**

- “Giá rẻ, săn sale rất hời.” → `1`
- “Chất lượng vậy mà bán quá đắt.” → `-1`
- “Giá niêm yết là 250 nghìn.” → `0`
- “Giá tốt nhưng voucher không áp được nên tổng tiền vẫn cao.” → `"1, -1"`

### 4.5. Vận chuyển

**Bao gồm:**

- thời gian giao, giao nhanh/chậm, trễ hẹn;
- phí ship;
- thái độ và thao tác của shipper;
- trạng thái hành trình giao nhận khi review trực tiếp đánh giá quá trình này.

**Không bao gồm:**

- cách bọc/hộp/chống sốc: dùng `Đóng gói`;
- tư vấn, phản hồi hoặc hành xử của shop;
- giao sai màu/size/mẫu: dùng `Đúng mô tả`;
- sản phẩm trầy/hỏng nếu review không quy nguyên nhân cho vận chuyển.

**Ví dụ:**

- “Giao nhanh, shipper thân thiện.” → `1`
- “Đơn trễ bốn ngày, shipper khó chịu.” → `-1`
- “Đơn được giao vào thứ Hai.” → `0`
- “Giao nhanh nhưng phí ship quá cao.” → `"1, -1"`

### 4.6. Đóng gói

**Bao gồm:**

- cách bọc hàng, túi, hộp, xốp, chống sốc và niêm phong;
- độ chắc chắn/cẩn thận của gói hàng;
- hộp hoặc bao bì vận chuyển bị móp, rách, ướt;
- khả năng đóng gói bảo vệ sản phẩm.

**Không bao gồm:**

- lỗi ngoại quan của sản phẩm khi không có bằng chứng liên quan đến đóng gói;
- giao chậm/nhanh;
- hộp sản phẩm như một bộ phận có tính năng, nếu review rõ ràng đang đánh giá
  chất lượng bản thân sản phẩm thay vì bao gói vận chuyển.

**Ví dụ:**

- “Bọc ba lớp xốp, rất cẩn thận.” → `1`
- “Không có chống sốc, hộp rách.” → `-1`
- “Sản phẩm được để trong hộp giấy.” → `0`
- “Bọc kỹ nhưng hộp vẫn bị móp.” → `"1, -1"`

### 4.7. Dịch vụ & Thái độ Shop

**Bao gồm:**

- tư vấn trước/trong khi mua;
- tốc độ và chất lượng phản hồi của shop;
- thái độ, sự chu đáo, trung thực hoặc hợp tác của người bán;
- xử lý yêu cầu của khách nếu chưa thuộc quy trình hậu mãi cụ thể.

**Không bao gồm:**

- thái độ shipper;
- thao tác tự động của sàn;
- bảo hành, đổi trả và hoàn tiền sau mua khi đó là trọng tâm: dùng
  `Bảo hành & Đổi trả`.

**Ví dụ:**

- “Shop trả lời nhanh, tư vấn đúng nhu cầu.” → `1`
- “Nhắn ba ngày không phản hồi.” → `-1`
- “Shop có nhắn xác nhận đơn.” → `0`
- “Tư vấn nhiệt tình nhưng sau khi chốt đơn thì trả lời rất khó chịu.” →
  `"1, -1"`

### 4.8. Bảo hành & Đổi trả

**Bao gồm:**

- chính sách và thời hạn bảo hành;
- yêu cầu đổi/trả sản phẩm;
- hoàn tiền, khiếu nại và hỗ trợ sau mua;
- trải nghiệm thực tế khi sản phẩm lỗi cần hậu mãi.

**Không bao gồm:**

- tư vấn bán hàng thông thường;
- lỗi sản phẩm tự thân nếu review không đề cập việc bảo hành/đổi trả;
- giao nhầm hàng nếu chưa nói đến quá trình đổi trả.

**Ví dụ:**

- “Shop đổi máy lỗi trong ngày, hỗ trợ rất nhanh.” → `1`
- “Từ chối bảo hành dù còn thời hạn.” → `-1`
- “Sản phẩm có bảo hành 12 tháng.” → `0`
- “Đồng ý đổi hàng nhanh nhưng hoàn tiền quá chậm.” → `"1, -1"`

### 4.9. Tính xác thực

**Bao gồm:**

- hàng chính hãng, real, authentic;
- hàng fake, nhái, giả;
- tem/chứng cứ xác thực khi review dùng chúng để kết luận nguồn gốc.

**Không bao gồm:**

- sản phẩm khác hình hoặc giao nhầm nhưng không có cáo buộc giả;
- sản phẩm chất lượng thấp nhưng không có bằng chứng về real/fake;
- câu hỏi không có kết luận về chính hãng phải được gán neutral, không suy đoán.

**Ví dụ:**

- “Check mã ra hàng chính hãng.” → `1`
- “Hàng fake, logo sai và không check được serial.” → `-1`
- “Không biết có phải hàng chính hãng không.” → `0`
- “Tem check được nhưng chất liệu khiến mình nghi là hàng nhái.” →
  `"1, -1"` nếu cả hai bằng chứng đều được người viết khẳng định trong review

## 5. Quy tắc suy luận chung

### 5.1. Chỉ gán khi có bằng chứng trong văn bản

Annotator phải xác định từ/cụm từ/mệnh đề làm evidence trước khi gán `-1`, `0`,
`1` hoặc `"1, -1"`. Evidence có thể:

- gọi thẳng tên khía cạnh;
- miêu tả thuộc tính thuộc khía cạnh;
- biểu đạt một category ngầm nhưng có đích rõ.

Ví dụ “rất đáng tiền” không nói từ “giá” nhưng có đích rõ là value, vì vậy
`Giá cả & Khuyến mãi=1`. Ngược lại, câu chỉ có “rất hài lòng” không xác định
khía cạnh nào trong chín nhóm, nên không được tự động gán tất cả positive hoặc
gán quality positive.

Quy ước cho đánh giá chung:

- “Hàng tốt”, “sản phẩm đẹp”, “áo xịn” có đích là sản phẩm →
  `Chất lượng sản phẩm=1`.
- “Shop tốt” → `Dịch vụ & Thái độ Shop=1`.
- “Rất tốt”, “ưng lắm”, “5 sao” nhưng không có đích → không suy ra một trong
  chín aspect; nếu mẫu còn tồn tại sau cleaning, gán `2` cho aspect không có
  evidence và đánh dấu `INSUFFICIENT_CONTEXT`.

### 5.2. Neutral không phải “không chắc nên chọn đại”

Nhãn `0` dùng khi văn bản thực sự nhắc đến khía cạnh nhưng:

- chỉ cung cấp thông tin thực tế;
- thể hiện dự định, mong muốn hoặc điều kiện mà chưa đánh giá;
- đặt câu hỏi hoặc nói chưa đủ trải nghiệm để kết luận;
- dùng mô tả không mang định hướng tích cực/tiêu cực trong ngữ cảnh.

Ví dụ:

- “Bảo hành 12 tháng.” → `Bảo hành & Đổi trả=0`
- “Chưa dùng nên chưa biết pin có bền không.” →
  `Hiệu năng & Trải nghiệm=0`
- “Không biết hàng có chính hãng không.” → `Tính xác thực=0`

Nếu khía cạnh không xuất hiện về mặt ngữ nghĩa, dùng `2`, không dùng `0`.

### 5.3. Phủ định và phạm vi

Phải đọc cả cụm phủ định, không gán theo từng từ riêng:

- “không chậm” → vận chuyển positive;
- “không tốt” → aspect tương ứng negative;
- “không hề ọp ẹp” → chất lượng positive;
- “không chỉ đẹp mà còn chắc” → chất lượng positive, không mixed.

Nếu phạm vi phủ định không rõ, gán nhãn tạm thời hợp lý nhất và đánh dấu
`POLARITY_SCOPE`.

### 5.4. Mức độ, so sánh và kỳ vọng

- Các từ “hơi”, “rất”, “cực kỳ” thay đổi cường độ nhưng không thay đổi loại
  nhãn.
- “Tốt hơn mong đợi” là positive.
- “Không tốt bằng hình” có thể đồng thời là `Đúng mô tả=-1` và
  `Chất lượng sản phẩm=-1` nếu cả hai ý đều được phát biểu.
- “Tạm được”, “bình thường” thường là neutral khi không có tín hiệu hài
  lòng/chê rõ. “Tạm được so với giá” có thể là giá positive nếu ngữ cảnh thể
  hiện chấp nhận giá trị.

### 5.5. Emoji, tiếng lóng và sarcasm

Emoji chỉ hỗ trợ polarity khi nó đi cùng một aspect có đích rõ:

- “Đóng gói kỹ ❤️” → đóng gói positive.
- “Pin tụt nhanh 🙂” có thể là sarcasm, nhưng không kết luận chỉ dựa vào emoji;
  evidence “tụt nhanh” đã cho hiệu năng negative.

Review chỉ có emoji không đủ để gán polarity cho một aspect. Sarcasm chỉ được
gán theo nghĩa ngữ dụng khi có tín hiệu ngôn ngữ rõ; nếu còn tranh cãi, dùng
`SARCASM`.

### 5.6. Đồng tham chiếu và nhiều câu

Phải đọc toàn bộ review. Đại từ hoặc từ lược bỏ có thể kế thừa đích rõ từ câu
trước:

> “Pin ban đầu khá trâu. Sau một tuần thì nó tụt rất nhanh.”

Hai mệnh đề đều nói về pin, nên `Hiệu năng & Trải nghiệm="1, -1"`.

Không gộp polarity của hai aspect khác nhau chỉ vì chúng nằm cùng một câu.

## 6. Multi-polarity `"1, -1"`

### 6.1. Điều kiện bắt buộc

Gán `"1, -1"` khi và chỉ khi:

1. có ít nhất một evidence positive;
2. có ít nhất một evidence negative;
3. hai evidence cùng thuộc **một** aspect;
4. cả hai đều là nhận xét có nghĩa, không phải do parser lặp lại cùng một ý.

Mixed có thể xuất hiện trong một câu hoặc ở hai câu khác nhau trong cùng review.
Không được chọn một polarity “chiếm ưu thế” theo số lượng từ, vị trí cuối câu
hoặc rating.

### 6.2. Những trường hợp không phải mixed

- Quality positive và price negative → hai aspect đơn polarity.
- Một mệnh đề neutral và một mệnh đề positive cùng aspect → `1`.
- Hai lời khen khác nhau cùng aspect → `1`.
- “Không xấu” → thường positive hoặc neutral theo ngữ cảnh, không tự động mixed.
- Người viết trích dẫn lời shop rồi bác bỏ nó không tạo hai quan điểm của người
  viết; phải xác định stance thực.

### 6.3. Ví dụ đối chiếu

| Review | Nhãn đúng |
|---|---|
| “Máy mượt nhưng thỉnh thoảng bị lag.” | Hiệu năng=`"1, -1"` |
| “Vải đẹp nhưng giá đắt.” | Quality=`1`; Price=`-1` |
| “Hộp giấy, được bọc rất kỹ.” | Packaging=`1`, không mixed |
| “Bọc kỹ nhưng hộp vẫn rách.” | Packaging=`"1, -1"` |
| “Giao nhanh, sản phẩm bị trầy.” | Shipping=`1`; Quality=`-1` |
| “Shop tư vấn tốt nhưng từ chối đổi hàng lỗi.” | Service=`1`; Warranty/Return=`-1` |

## 7. Các ranh giới dễ nhầm

| Cặp khía cạnh | Câu hỏi phân biệt |
|---|---|
| Quality – Performance | Đang nói thuộc tính/hình thể của sản phẩm hay cách nó hoạt động/trải nghiệm khi dùng? |
| Quality – Description | Đang đánh giá bản thân thuộc tính hay so nó với listing/đơn đã đặt? |
| Shipping – Packaging | Đang nói thời gian/shipper/phí hay vật liệu/cách bọc/hộp? |
| Shipping – Shop Service | Đối tượng bị đánh giá là shipper/quá trình giao hay nhân viên/người bán? |
| Shop Service – Warranty/Return | Tư vấn bán hàng chung hay quy trình hậu mãi cụ thể? |
| Description – Authenticity | Khác thông tin/hình/biến thể hay cáo buộc real/fake? |
| Price – Quality | “Đắt/rẻ/đáng tiền” thuộc price; lỗi/bền/đẹp thuộc quality. Một câu có thể gán cả hai. |

Khi một mệnh đề thực sự chứa hai ý độc lập, được phép gán hai aspect. Mục đích
không phải ép mỗi mệnh đề vào đúng một cột, mà là tránh suy diễn một ý sang cột
không có evidence.

## 8. Xử lý nội dung không phải review

Các dấu hiệu cần trả mẫu về bước data cleaning:

- quảng cáo shop/livestream hoặc lời kêu gọi mua hàng không phải trải nghiệm
  của người mua;
- link chia sẻ sản phẩm, mã giới thiệu;
- OTP, SMS, thông báo hệ thống, nội dung bàn phím/Gboard;
- tin tức, văn bản pháp luật hoặc nội dung ngoài thương mại điện tử;
- chuỗi template nhận xu không có bất kỳ nhận xét sản phẩm/dịch vụ nào;
- văn bản hỏng, chỉ ký tự rác hoặc không đủ ngữ nghĩa.

Annotator đặt `annotation_status=REJECT_NON_REVIEW`, chọn mã
`NON_REVIEW`, và ghi lý do ngắn. Không điền chín nhãn final cho mẫu bị reject.
Nhóm quản trị dữ liệu quyết định loại hay khôi phục mẫu và phải lưu quyết định
trong exclusion ledger.

Nếu review gồm cả boilerplate và nhận xét thật, annotator không tự xóa chữ.
Họ đánh dấu `BOILERPLATE_MIXED_WITH_REVIEW`, xác định evidence thật và gửi kiểm
duyệt. Nhãn chỉ được giữ trong gold dataset sau khi nhóm cleaning xác nhận phiên
bản text canonical.

## 9. Quy trình gán nhãn một review

Annotator thực hiện tuần tự:

1. Đọc toàn bộ review một lần để hiểu chủ thể và ngữ cảnh.
2. Kiểm tra nó có phải review thực hay cần `REJECT_NON_REVIEW`.
3. Đọc lại và đánh dấu các evidence clause.
4. Ánh xạ từng evidence sang một hoặc nhiều aspect theo định nghĩa ở Mục 4.
5. Xác định polarity độc lập cho từng aspect.
6. Gộp evidence ở cấp toàn review:
   - chỉ positive → `1`;
   - chỉ negative → `-1`;
   - chỉ neutral → `0`;
   - positive và negative → `"1, -1"`;
   - không có evidence → `2`.
7. Điền đủ chín ô nhãn.
8. Thêm mã không chắc chắn nếu có.
9. Kiểm tra lại: không lan sentiment, không bỏ sót mixed, không còn ô trống.
10. Gửi annotation.

Evidence audit nên lưu theo cấu trúc:

```text
sample_id
aspect
label
evidence_text
evidence_start
evidence_end
uncertainty_code
guideline_version
annotator_id
```

Offsets phải tham chiếu đúng `reviewContent` canonical. Evidence không phải là
thành phần bắt buộc của schema legacy, nhưng là artefact quan trọng để
adjudication và kiểm toán dataset.

## 10. Mã không chắc chắn và escalation

Các mã chuẩn:

| Mã | Khi sử dụng |
|---|---|
| `ASPECT_BOUNDARY` | Không chắc evidence thuộc aspect nào |
| `POLARITY_SCOPE` | Phạm vi phủ định/cảm xúc không rõ |
| `SARCASM` | Có khả năng mỉa mai/châm biếm |
| `INSUFFICIENT_CONTEXT` | Văn bản có nghĩa nhưng không đủ xác định aspect/polarity |
| `TYPO_LANGUAGE` | Lỗi gõ, tiếng lóng hoặc ngôn ngữ pha trộn gây khó hiểu |
| `NON_REVIEW` | Nội dung có vẻ không phải review |
| `BOILERPLATE_MIXED_WITH_REVIEW` | Có review thật lẫn template/spam |
| `OTHER` | Ca khác, phải kèm ghi chú |

Trừ `NON_REVIEW`, annotator vẫn phải chọn nhãn tạm thời; không để blank chỉ vì
không chắc. Các mẫu có mã uncertainty được đưa vào adjudication bắt buộc.

## 11. Ví dụ đầy đủ theo vector chín khía cạnh

Thứ tự vector luôn là:

```text
[Quality, Performance, Description, Price, Shipping,
 Packaging, ShopService, WarrantyReturn, Authenticity]
```

### Ví dụ A

> “Shop giao hàng siêu nhanh, bọc xốp cẩn thận nhưng hộp vẫn bị móp.”

```text
[2, 2, 2, 2, 1, "1, -1", 2, 2, 2]
```

### Ví dụ B

> “Điện thoại cầm đầm tay, màn đẹp nhưng pin tụt nhanh, giá quá đắt.”

```text
[1, -1, 2, -1, 2, 2, 2, 2, 2]
```

### Ví dụ C

> “Tư vấn nhiệt tình, đặt áo xanh mà giao áo đỏ nhưng mặc vẫn đẹp.”

```text
[1, 2, -1, 2, 2, 2, 1, 2, 2]
```

### Ví dụ D

> “Chưa dùng nên chưa biết máy chạy ra sao. Hàng giao đúng hình.”

```text
[2, 0, 1, 2, 2, 2, 2, 2, 2]
```

### Ví dụ E

> “Check serial ra chính hãng nhưng màu khác hẳn ảnh quảng cáo.”

```text
[2, 2, -1, 2, 2, 2, 2, 2, 1]
```

### Ví dụ F

> “Shop đồng ý đổi rất nhanh nhưng một tuần vẫn chưa hoàn tiền.”

```text
[2, 2, 2, 2, 2, 2, 2, "1, -1", 2]
```

## 12. Tổ chức đội gán nhãn

### 12.1. Vai trò tối thiểu

- **Hai annotator chính:** gán độc lập 100% mẫu.
- **Một expert adjudicator:** giải quyết bất đồng và ca uncertainty.
- **Một data curator:** quản lý text/hash, mẫu bị reject và version dataset.

Expert adjudicator không được tính là “annotator độc lập thứ ba” nếu họ đã nhìn
thấy nhãn A/B. Nếu cần báo IAA ba người, annotator thứ ba phải gán một subset
blind trước khi được xem bất kỳ nhãn nào khác.

### 12.2. Đào tạo và pilot

Quy trình bắt buộc:

1. Học taxonomy và thảo luận ví dụ chuẩn.
2. Làm vòng practice có feedback.
3. Làm ít nhất hai vòng pilot độc lập, tổng khoảng 100–200 review.
4. Pilot phải được lấy stratified để có:
   - đủ chín aspect;
   - absent và neutral;
   - positive, negative và mixed;
   - các cặp aspect dễ nhầm;
   - lỗi chính tả, phủ định, sarcasm và multi-sentence.
5. Sửa guideline/codebook nếu bất đồng có tính hệ thống.
6. Nếu rule thay đổi, gán lại toàn bộ pilot bằng phiên bản mới.

Gate đề xuất để vào production:

- mention macro-F1 so với expert gold từ `0.85` trở lên;
- full 5-state macro-F1 từ `0.80` trở lên;
- exact agreement với gold từ `0.85` trở lên;
- annotator giải thích đúng các ca mixed và neutral-vs-absent.

Ngưỡng này là operational quality gate và phải được đăng ký trước khi chạy
main annotation; không điều chỉnh ngưỡng sau khi xem kết quả để làm đẹp báo cáo.

## 13. Gán nhãn production: double-blind

Toàn bộ mẫu chính thức phải được hai annotator gán:

- độc lập;
- ở thứ tự ngẫu nhiên;
- không trao đổi về mẫu đang làm;
- không nhìn nhãn của nhau;
- không nhìn model suggestion;
- không nhìn metadata bị cấm ở Mục 2.2.

Nên dùng batch 300–500 review và giới hạn phiên làm việc để giảm fatigue.
Khoảng 5% hidden expert-gold và 2% repeated controls có thể được chèn vào luồng
annotation. Các bản lặp kiểm soát không được tính thành review mới trong
dataset.

Mỗi batch phải qua validation:

- đúng số lượng sample;
- `review_text_sha256` không đổi;
- đủ chín nhãn hợp lệ cho mọi mẫu đã hoàn tất;
- không có biểu diễn mixed sai format;
- không có cột hoặc thứ tự cột sai;
- hidden-gold và intra-annotator consistency đạt gate đã đăng ký.

Không tự động loại annotator chỉ dựa vào thời gian thao tác. Tốc độ bất thường
là tín hiệu để mở audit. Nếu một annotator không đạt batch gate, phải tạm dừng
và gán lại các batch kể từ quality checkpoint gần nhất.

## 14. Inter-Annotator Agreement (IAA)

### 14.1. Thời điểm tính

IAA phải được tính trên hai bộ nhãn **độc lập, trước adjudication**. Không tính
IAA trên nhãn final vì hòa giải làm agreement tăng giả tạo.

### 14.2. Các thước đo bắt buộc

Không dùng một con số kappa duy nhất. Báo ít nhất:

1. **Full five-state task** `{2, -1, 0, 1, mixed}`:
   - raw exact agreement;
   - Gwet's AC1 hoặc Krippendorff's alpha nominal;
   - Cohen's kappa có thể báo bổ sung cho hai annotator.
2. **Mention detection** `{absent, mentioned}`:
   - agreement;
   - precision, recall, F1 theo từng aspect;
   - macro-F1 và AC1.
3. **Polarity conditional** trên những cặp mà cả hai annotator đều xác định
   aspect được nhắc:
   - exact agreement;
   - alpha/AC1 cho bốn trạng thái `negative`, `neutral`, `positive`, `mixed`;
   - confusion matrix.
4. **Set-valued polarity**:
   - exact-set match;
   - Jaccard/F1 trên tập `{negative, positive, neutral}`;
   - mixed-specific precision, recall, F1 và support.

Phải báo per-aspect, micro, macro và số lượng support. Do absent thường áp đảo,
raw accuracy hoặc kappa có thể gây hiểu nhầm; Gwet AC1, F1 và confusion matrix
giúp giải thích prevalence.

Khoảng tin cậy 95% nên bootstrap theo `product/template group`, không coi từng
review gần trùng là quan sát iid.

### 14.3. Quality gate IAA đề xuất

- mention macro-F1 ≥ `0.85`;
- mention AC1 ≥ `0.80`;
- full five-state exact agreement ≥ `0.80`;
- polarity alpha/AC1 overall ≥ `0.80`;
- không aspect có support đủ lớn nào thấp hơn `0.67`.

Mức `0.67–0.80` yêu cầu audit và cải thiện guideline. Dưới `0.67` yêu cầu dừng
vòng annotation liên quan, phân tích lỗi và gán lại. Mixed thường hiếm nên phải
báo support; mọi bất đồng mixed phải adjudicate, không che bằng overall score.

## 15. Adjudication

Expert adjudicator phải xem:

- review canonical;
- evidence của A và B;
- hai bộ nhãn được hiển thị ở thứ tự ngẫu nhiên;
- uncertainty code;
- guideline version có hiệu lực.

Expert không được thấy danh tính annotator khi ra quyết định. Các trường hợp
bắt buộc adjudicate:

- mọi bất đồng ở bất kỳ aspect nào;
- mọi mẫu có uncertainty code;
- mọi mẫu `NON_REVIEW` hoặc `BOILERPLATE_MIXED_WITH_REVIEW`;
- toàn bộ ca mixed hiếm trong giai đoạn đầu;
- một mẫu ngẫu nhiên 5–10% từ nhóm A/B đã đồng thuận để audit.

Quyết định không được dựa trên majority vote máy móc. Expert phải chọn nhãn có
bằng chứng phù hợp guideline và lưu:

```text
sample_id
aspect
annotator_A_label
annotator_B_label
final_label
reason_code
adjudication_note
adjudicator_id
guideline_version
adjudicated_at
```

Nếu adjudication phát hiện một quy tắc mới:

1. cập nhật và tăng phiên bản guideline;
2. ghi changelog;
3. truy vấn toàn bộ mẫu có thể bị ảnh hưởng;
4. gán lại/audit hồi tố các mẫu đó;
5. không áp dụng quy tắc mới chỉ cho dữ liệu về sau.

## 16. Kiểm tra chất lượng bản nhãn cuối

Một release được phép dùng cho ML/DL chỉ khi:

- mỗi mẫu có đúng chín nhãn final;
- số ô nhãn blank bằng `0`;
- mọi giá trị thuộc `{2, -1, 0, 1, "1, -1"}`;
- mọi mixed có đúng canonical format;
- tensor mapping giữ thứ tự `[negative, positive, neutral]`;
- mọi disagreement có bản ghi adjudication;
- mọi uncertainty đã được resolve;
- mọi mẫu `NON_REVIEW` đã được loại hoặc có quyết định curator rõ;
- `sample_id` là duy nhất;
- `review_text_sha256` khớp text đã annotation;
- không có annotation control bị tính như một sample mới;
- có bảng phân bố label theo aspect và polarity;
- có IAA report tính trước adjudication;
- guideline version của từng dòng được lưu trong provenance.

Trước training cần có một validator strict. Validator phải dừng với lỗi, không
tự sửa, khi gặp blank, nhãn lạ, cột thiếu hoặc text hash sai.

## 17. Quản lý phiên bản và khả năng tái lập

Mỗi lần sửa guideline phải ghi:

- số phiên bản;
- ngày;
- quy tắc thay đổi;
- lý do;
- nhóm sample bị ảnh hưởng;
- hành động hồi tố;
- người phê duyệt.

Các artefact annotation tối thiểu phải được giữ:

- guideline và changelog;
- taxonomy/schema machine-readable;
- pilot/gold set;
- assignment manifest;
- hai bộ nhãn raw độc lập;
- evidence và uncertainty;
- IAA script, input và report;
- adjudication ledger;
- danh sách mẫu reject cùng reason;
- final labels và checksum;
- danh sách annotator dưới ID giả danh và hồ sơ training/qualification.

Nhãn do LLM hoặc model sinh có thể dùng làm thí nghiệm weak supervision riêng,
nhưng không được gọi là human gold và không được đưa cho annotator trong vòng
double-blind.

## 18. Checklist ngắn dành cho annotator

Trước khi bấm **Submit**, tự hỏi:

1. Đây có thực sự là review không?
2. Tôi đã đọc toàn bộ review, không chỉ câu đầu chưa?
3. Mỗi nhãn có evidence trong văn bản không?
4. Tôi có nhầm neutral `0` với absent `2` không?
5. Tôi có lan một lời khen/chê sang aspect khác không?
6. Có positive và negative cho cùng aspect cần `"1, -1"` không?
7. Tôi có bị rating, tên sản phẩm hoặc nhãn cũ ảnh hưởng không?
8. Đủ đúng chín ô và không còn blank chưa?
9. Ca không chắc đã có uncertainty code chưa?

Chỉ submit khi cả chín quyết định phản ánh đúng văn bản và đúng phiên bản
guideline hiện hành.
