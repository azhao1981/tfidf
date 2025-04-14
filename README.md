# go-tf-idf

go 实现的 TF-IDF 算法

```bash
git clone git@github.com:azhao1981/go-tf-idf.git
```

```go
package main

import (
 "fmt"
 "strings"
)

func main() {
 docs := [][]string{
  strings.Split("the cat sat on the mat", " "),
  strings.Split("the dog slept on the bed", " "),
  strings.Split("the cat chased the mouse", " "),
 }
 ad := NewArrayDocs(docs)

 // 计算特定词在第一个文档中的TF-IDF
 sentence := strings.Split("cat chased mouse", " ")
 ti := NewTFIDF(ad.docs)
 result := ti.Scan(sentence)
 fmt.Println(result)
 sdocs, scores := ti.SortedDocs(5)
 for i, doc := range sdocs {
  fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
 }

 docs := []TestDoc{
  {ID: 1, Text: "the cat sat on the mat"},
  {ID: 2, Text: "the dog slept on the bed"},
  {ID: 3, Text: "the cat chased the mouse"},
 }
 ti := NewTFIDF(docs)
 result := ti.ScanWithCoverage(strings.Split("cat chased mouse", " "), 0.3)
 fmt.Println(result)
 docs, scores := ti.SortedDocs(10)
 for i, doc := range docs {
  fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
}
```

```bash
go list -m  github.com/azhao1981/go-tf-idf@v1.0.3
```
