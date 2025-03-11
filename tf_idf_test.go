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
	time.Sleep(111 * time.Microsecond)
	docs := [][]string{
		strings.Split("the cat sat on the mat", " "),
		strings.Split("the dog slept on the bed", " "),
		strings.Split("the cat chased the mouse", " "),
	}

	// 计算特定词在第一个文档中的TF-IDF
	sentence := strings.Split("cat chased mouse", " ")
	ti := NewTFIDF(docs)
	result := ti.Scan(sentence)
	fmt.Println(result)
	for _, doc := range ti.SortedDocs() {
		fmt.Println(doc)
	}

	assert.True(t, false)
}
