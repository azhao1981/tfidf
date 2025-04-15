package tfidf

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/go-ego/gse"
	"github.com/stretchr/testify/assert"
)

func TestTFIDF(t *testing.T) {
	docs := [][]string{
		strings.Split("the cat sat on the mat", " "),
		strings.Split("the dog slept on the bed", " "),
		strings.Split("the cat chased the mouse", " "),
	}

	// 计算特定词在第一个文档中的TF-IDF
	sentence := strings.Split("cat chased mouse", " ")
	for _, doc := range docs {
		score := SentenceTFIDF(doc, docs, sentence)
		fmt.Printf("TF-IDF for '%s': %.4f %v+\n", sentence, score, doc)
	}
}

func TestTFIDF2(t *testing.T) {
	time.Sleep(1 * time.Microsecond)
	docs := [][]string{
		strings.Split("the cat sat on the mat", " "),
		strings.Split("the dog slept on the bed", " "),
		strings.Split("the cat chased the mouse", " "),
	}
	ad := NewArrayDocs(docs)

	// 计算特定词在第一个文档中的TF-IDF
	sentence := strings.Split("cat chased mouse", " ")
	ti := NewTFIDF(ad.Docs)
	result := ti.Scan(sentence)
	fmt.Println(result)
	sdocs, scores := ti.SortedDocs(5)
	for i, doc := range sdocs {
		fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
	}
}

type TestDoc struct {
	ID   int
	Text string
	Seg  string
}

func (t TestDoc) Words() []string {
	return strings.Split(t.Text, t.Seg)
}

func TestTFIDF3(t *testing.T) {
	time.Sleep(12 * time.Microsecond)
	docs := []TestDoc{
		{ID: 1, Text: "the cat sat on the mat", Seg: " "},
		{ID: 2, Text: "the dog slept on the bed", Seg: " "},
		{ID: 3, Text: "the cat chased the mouse", Seg: " "},
	}

	// 计算特定词在第一个文档中的TF-IDF
	sentence := strings.Split("cat chased mouse", " ")
	ti := NewTFIDF(docs)
	result := ti.Scan(sentence)
	fmt.Println(result)
	docs, scores := ti.SortedDocs(10)
	for i, doc := range docs {
		fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
	}

}

func TestTFIDFA(t *testing.T) {
	time.Sleep(12 * time.Microsecond)
	docs := []TestDoc{
		{ID: 0, Text: "广东省惠州市惠东县大岭街道东兴社区", Seg: ""},
		{ID: 1, Text: "广东省惠州市惠城区水口街道东兴社区", Seg: ""},
		{ID: 2, Text: "广东省惠州市惠阳区大亚湾霞涌街道东兴社区", Seg: ""},
	}

	// 计算特定词在第一个文档中的TF-IDF
	sentence := strings.Split(" 广东省惠州市大亚湾东兴社区", "")
	ti := NewTFIDF(docs)
	result := ti.ScanWithCoverage(sentence, 0.3)
	fmt.Println(result)
	ret := ti.QueryCoverage(sentence)
	fmt.Println("QueryCoverage", ret)
	CovScores := QueryCoverage(sentence, ti.docs)
	fmt.Println("CovScores", CovScores, "\nqueryCoverage", ti.queryCoverage)
	for i, score := range ti.finalScores {
		fmt.Printf("sentenceScore %d: %s score: %.4f, tf:%.4f, cov: %.4f\n",
			i, docs[score.DocIndex].Text, score.Score, ti.docIFIDFScores[score.DocIndex], ti.queryCoverage[score.DocIndex])
	}
	docs, scores := ti.SortedDocs(3)
	for i, doc := range docs {
		fmt.Printf("%d: %s %d,  score: %.4f, \n", i, doc.Text, doc.ID, scores[i])
	}
	fmt.Println(ti.queryCoverage)

}
func TestCoverRate(t *testing.T) {
	docs := []TestDoc{
		{ID: 1, Text: "the cat sat on the mat", Seg: " "},
		{ID: 2, Text: "the dog slept on the bed", Seg: " "},
		{ID: 3, Text: "the cat chased the mouse", Seg: " "},
	}
	doc := strings.Split("cat chased mouse", " ")
	rates := QueryCoverage(doc, docs)
	for i, rate := range rates {
		fmt.Printf("%d: %v\n", i, rate)
	}
	assert.Equal(t, 0.3333333333333333, rates[0])
	assert.Equal(t, 0.0, rates[1])
	assert.Equal(t, 1.0, rates[2])
}

func TestScanWithCoverage(t *testing.T) {
	docs := []TestDoc{
		{ID: 1, Text: "the cat sat on the mat", Seg: " "},
		{ID: 2, Text: "the dog slept on the bed", Seg: " "},
		{ID: 3, Text: "the cat chased the mouse", Seg: " "},
	}
	ti := NewTFIDF(docs)
	result := ti.ScanWithCoverage(strings.Split("cat chased mouse", " "), 0.3)
	fmt.Println(result)
	for _, score := range ti.finalScores {
		fmt.Printf("%d: %v, score: %.4f\n", score.DocIndex, docs[score.DocIndex], score.Score)
	}
	docs, scores := ti.SortedDocs(10)
	for i, doc := range docs {
		fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
	}
}

func TestScanWithCoverageNaN(t *testing.T) {
	docs := []TestDoc{
		{ID: 3, Text: "the cat chased the mouse", Seg: " "},
	}
	ti := NewTFIDF(docs)
	result := ti.ScanWithCoverage(strings.Split("the cat chased mouse", " "), 0.3)
	fmt.Println(result)
	for _, score := range ti.finalScores {
		fmt.Printf("sentenceScore %d: %v, score: %.4f\n", score.DocIndex, docs[score.DocIndex], score.Score)
	}
	ti.NormalizeSentenceScore()
	docs, scores := ti.SortedDocs(10)
	for i, doc := range docs {
		fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
	}
}

func TestScanWithCoverageNaN2(t *testing.T) {
	docs := []TestDoc{
		{ID: 3, Text: "the cat chased the mouse", Seg: " "},
	}
	ti := NewTFIDF(docs)
	result := ti.ScanWithCoverage(strings.Split("the dog slept on the bed", " "), 0.3)
	fmt.Println(result)
	for _, score := range ti.finalScores {
		fmt.Printf("sentenceScore %d: %v, score: %.4f\n", score.DocIndex, docs[score.DocIndex], score.Score)
	}
	ti.NormalizeSentenceScore()
	docs, scores := ti.SortedDocs(10)
	for i, doc := range docs {
		fmt.Printf("%d: %v, score: %.4f\n", i, doc, scores[i])
	}
}

func TextToArray(text string) []string {
	lines := strings.Split(strings.TrimSpace(text), "\n")
	lines = removeEmptyLines(lines)
	return lines
}
func removeEmptyLines(lines []string) []string {
	nonEmptyLines := make([]string, 0)
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			nonEmptyLines = append(nonEmptyLines, line)
		}
	}
	return nonEmptyLines
}

type TestGseDoc struct {
	ID   int
	Text string
	Seg  gse.Segmenter
}

func (t TestGseDoc) Words() []string {
	return t.Seg.Cut(t.Text, true)
}

func TestNav(t *testing.T) {
	text := `
	海南省海口市秀英区游2路-海南野生动植物园
海南省海口市秀英区生态休闲小道海南热带野生动植物园-海野恐龙园
海南省海口市秀英区琼东路海南热带野生动植物园-售票处
海南省海口市秀英区东山镇琼东路688号-海南热带野生动植物园
海南省海口市秀英区东山镇(海榆中线27号公里处)海南热带野生动植物园内-南药园
海南省海口市秀英区东山镇(海榆中线27号公里处)-海南热带野生动植物园-东门
海南省海口市秀英区东山镇(海榆中线27号公里处)海南热带野生动植物园内-龙眼
海南省海口市秀英区东山镇(海榆中线27号公里处)海南热带野生动植物园-游客中心
海南省海口市秀英区东山镇(海榆中线27号公里处)海南热带野生动植物园-停车场内-海南热带野生动植物园P1地上停车场-出入口
海南省海口市秀英区东山镇(海榆中线27号公里处)-海南热带野生动植物园-P1地上停车场
`
	lines := TextToArray(text)
	for _, line := range lines {
		fmt.Println(line)
	}

	var seg gse.Segmenter
	seg.LoadDictEmbed()
	docs := []TestGseDoc{}
	for idx, line := range lines {
		docs = append(docs, TestGseDoc{ID: idx, Text: line, Seg: seg})
	}
	ti := NewTFIDF(docs)
	// result := ti.Scan(seg.Cut("海南省海口市海南野生动植物园", true))
	result := ti.ScanWithCoverage(seg.Cut("海南省海口市海南野生动植物园", true), 0.1)
	// ti.NormalizeSentenceScore()
	fmt.Println(result)
	docs, scores := ti.SortedDocs(10)
	for i, doc := range docs {
		fmt.Printf("%d: %v, score: %.4f, coverage: %.d\n", i, doc.Text, scores[i], doc.ID)
	}
}
