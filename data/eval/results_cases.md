# Các trường hợp lấy từ kết quả chạy thật

Ground truth gán ở mức trang PDF; cần duyệt lại đoạn trích để đánh giá độ đầy đủ của đáp án. Không có điểm chất lượng LLM tự bịa.

Chênh lệch chunking lớn nhất: K5 / bm25_only, Δ(P@3 + R@3)=1.3333.
- fixed: P@3=0.6667, R@3=0.6667
- recursive: P@3=0.0000, R@3=0.0000

## Kết quả tốt nhất

S5 · fixed · hybrid_rrf
Explain external conceptual and internal models and their relationship to data independence.

P@3=1.0000, R@3=0.3333

- Chunk 107, Chapter 5 Data Modelling > Data Abstraction Layer, PDF [19, 20]: this stage, the conceptual model is independent of both software and hardware. It does not depend on the DBMS software used to implement the model. It does not depend on the hardware used in the implementation of the model. Changes in either hardware or DBMS software have no effect on the database design at the conceptual level. Once a DBMS is selected, you can then implement it. This is the inter
- Chunk 105, Chapter 5 Data Modelling > Degrees of Data Abstraction, PDF [17, 18]: models in order of decreasing level of abstraction. External models • Represent the user’s view of the database • Contain multiple different external views • Are closely related to the real world as perceived by each user Conceptual models • Provide flexible data-structuring capabilities • Present a “community view”: the logical structure of the entire database • Contain data stored in the databas
- Chunk 109, Chapter 5 Data Modelling > Physical data independence, PDF [20, 21]: of the storage structure from user applications. The applications should not be involved with these issues, since there is no difference in the operation carried out against the data. Key Terms conceptual model: the logical structure of the entire database conceptual schema: another term for logical schema data independence: the immunity of user applications to changes made in the definition and o

## Kết quả yếu nhất / chưa đầy đủ

P1 · fixed · hybrid_rrf
Làm thế nào để nhiều người cùng truy cập một bản ghi mà không ảnh hưởng giao dịch?

P@3=0.0000, R@3=0.0000

- Chunk 8, Chương 5. Tập phát triển và tập kiểm tra > Điều gì đã xảy ra?, PDF [10]: đánh giá khả năng hoạt động của thuật toán một cách nhanh chóng. Nói cách khác, mục đích của tập phát triển và tập kiểm tra là hướng nhóm của bạn tới những thay đổi quan trọng nhất có thể làm để cải thiện trong hệ thống học máy. Vì vậy, bạn nên làm những điều sau đây: Lựa chọn tập phát triển và tập kiểm tra để phản ánh đúng dữ liệu bạn sẽ gặp phải trong tương lai và muốn hoạt động tốt trên nó. Nói
- Chunk 61, Chương 24. Đánh đổi giữa Độ chệch và Phương sai, PDF [42, 43]: sai. Ví dụ việc tăng kích thước mô hình -- như thêm các nơ-ron/tầng trong mạng nơ-ron hoặc thêm các đầu vào đặc trưng -- nhìn chung sẽ giảm độ chệch nhưng có thể làm tăng phương sai. Ngoài ra, việc thêm điều chuẩn thường làm tăng độ chệch nhưng giảm phương sai. Ngày nay, chúng ta có thể truy cập vào nguồn dữ liệu phong phú và có thể sử dụng các mạng nơ-ron rất lớn (trong học sâu). Vì thế mà ta ít 
- Chunk 81, Chương 33. Tại sao cần so sánh chất lượng mức con người?, PDF [57, 58]: mức hợp lý sẽ giúp đẩy nhanh tiến độ của nhóm. Việc biết thuật toán của bạn có độ chệch cao có thể tránh được là vô cùng có giá trị và mở ra nhiều tùy chọn để thử nghiệm. Có những tác vụ mà ngay cả con người cũng làm không tốt. Ví dụ, chọn một cuốn sách để giới thiệu cho bạn; hoặc chọn một tin quảng cáo để hiển thị cho người dùng trên một trang web; hoặc dự đoán thị trường chứng khoán. Máy tính đã

## Đối sánh chunking A

K5 · fixed · bm25_only
What are the properties of a database?

P@3=0.6667, R@3=0.6667

- Chunk 92, Chapter 1 Before the Advent of Database Systems > Key Terms, PDF [4, 5, 6]: manipulate and retrieve data in a variety of ways file-based system: an application program designed to manipulate data files Exercises 1. Discuss each of the following terms: 1. data 2. field 3. record 4. file 2. What is data redundancy? 3. Discuss the disadvantages of file-based systems. 4. Explain the difference between data and information. 5. Use Figure 1.2 (below) to answer the following que
- Chunk 94, Chapter 2 Fundamental Concepts > Database Management System, PDF [7, 8, 9]: is a collection of programs that enables users to create and maintain databases and control all access to them. The primary goal of a DBMS is to provide an environment that is both convenient and efficient for users to retrieve and store information. With the database approach, we can have the traditional banking system as shown in Figure 2.3. In this bank example, a DBMS is used by the Personnel 
- Chunk 118, Chapter 7 The Relational Data Model > Properties of a Table, PDF [30, 31, 32]: ◦ date ◦ logical (true or false) • Operations combining different data types are disallowed. • Each attribute has a distinct name. • The sequence of columns is insignificant. • The sequence of rows is insignificant. Key Terms atomic value: each value in the domain is indivisible as far as the relational model is concernedattribute: principle storage unit in a database column: see attribute degree:

## Đối sánh chunking B

K5 · recursive · bm25_only
What are the properties of a database?

P@3=0.0000, R@3=0.0000

- Chunk 98, Chapter 2 Fundamental Concepts > Exercises, PDF [8]: Exercises 1. What is a database management system (DBMS)?  2. What are the properties of a DBMS?  3. Provide three examples of a real-world database (e.g., the library contains a database of books).
- Chunk 159, Chapter 7 The Relational Data Model > Exercises, PDF [31]: Exercises Use Table 7.2 to answer questions 1-4.  1. Using correct terminology, identify and describe all the components in Table 7.2.  2. What is the possible domain for field EmpJobCode?  3. How many records are shown?  4. How many attributes are shown?  5. List the properties of a table.  Table 7.2. Table for exercise questions, by A. Watt.
- Chunk 156, Chapter 7 The Relational Data Model > Properties of a Table, PDF [30]: Properties of a Table • A table has a name that is distinct from all other tables in the database.  • There are no duplicate rows; each row is distinct.  • Entries in columns are atomic. The table does not contain repeating groups or multivalued attributes.  • Entries from columns are from the same domain based on their data type including:  ◦ number (numeric, integer, float, smallint,…) ◦ charact
