package tfidf

import (
	"fmt"
	"strings"
	"testing"
	"time"

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

	assert.True(t, false)
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
}

func (t TestDoc) Words() []string {
	return strings.Split(t.Text, " ")
}

func TestTFIDF3(t *testing.T) {
	time.Sleep(12 * time.Microsecond)
	docs := []TestDoc{
		{ID: 1, Text: "the cat sat on the mat"},
		{ID: 2, Text: "the dog slept on the bed"},
		{ID: 3, Text: "the cat chased the mouse"},
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
