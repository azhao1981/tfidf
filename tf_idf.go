package tfidf

import (
	"errors"
	"math"
	"sort"
)

type Tokenizable interface {
	Words() []string
}

type TFIDF[T Tokenizable] struct {
	docs          []T
	Sentence      []string
	termIDFMap    map[string]float64         // 每个词在文档中的idf值
	termTFMap     map[string]map[int]float64 // 每个词在每个文档中的tf值
	sentenceScore []*SentenceScore           // 句子对于每个文档的得分
	queryCoverage []float64                  // 查询对于每个文档的覆盖率
}

type SentenceScore struct {
	DocIndex int
	Score    float64
}

type Docs struct {
	Docs []Doc
}

func NewArrayDocs(docs [][]string) *Docs {
	adocs := make([]Doc, 0)
	for _, doc := range docs {
		ad := Doc{
			Items: doc,
		}
		adocs = append(adocs, ad)
	}
	return &Docs{
		Docs: adocs,
	}
}

type Doc struct {
	Items []string
}

func (doc Doc) Words() []string {
	return doc.Items
}

func NewTFIDF[T Tokenizable](docs []T) *TFIDF[T] {
	return &TFIDF[T]{
		docs:       docs,
		termIDFMap: make(map[string]float64),
		termTFMap:  make(map[string]map[int]float64),
	}
}

func (ti *TFIDF[T]) Scan(sentence []string) []string {
	ti.ScanDocs(sentence)
	return ti.ScanSentence(sentence)
}

func (ti *TFIDF[T]) ScanWithCoverage(sentence []string, weight float64) []string {
	ti.ScanDocs(sentence)
	ti.QueryCoverage(sentence)
	ti.NormalizeSentenceScore()
	for i, score := range ti.sentenceScore {
		ti.sentenceScore[i].Score = score.Score*(1-weight) + ti.queryCoverage[score.DocIndex]*weight
	}
	return ti.ScanSentence(sentence)
}

func (ti *TFIDF[T]) NormalizeSentenceScore() {
	scores := make([]float64, len(ti.sentenceScore))
	for i, score := range ti.sentenceScore {
		scores[i] = score.Score
	}
	scoresNormalized, err := ZScoreStandardize(scores)
	if err != nil {
		return
	}
	for i, _ := range ti.sentenceScore {
		ti.sentenceScore[i].Score = scoresNormalized[i]
	}
}

func (ti *TFIDF[T]) QueryCoverage(query []string) []float64 {
	CovScores := QueryCoverage(query, ti.docs)
	// 归一化
	CovScoresNormalized, err := ZScoreStandardize(CovScores)
	if err != nil {
		copy(ti.queryCoverage, CovScores)
		return CovScores
	}
	copy(ti.queryCoverage, CovScoresNormalized)
	return CovScoresNormalized
}

func (ti *TFIDF[T]) SortedDocs(k int) ([]T, []float64) {
	if k > len(ti.sentenceScore) {
		k = len(ti.sentenceScore)
	}
	docs := make([]T, k)
	scores := make([]float64, k)
	for i, doc := range ti.sentenceScore[:k] {
		docs[i] = ti.docs[doc.DocIndex]
		scores[i] = doc.Score
	}
	return docs, scores
}

func (ti *TFIDF[T]) ScanSentence(sentence []string) []string {
	ti.Sentence = sentence
	// 初始化 termScore
	ti.sentenceScore = make([]*SentenceScore, 0)
	ti.ScanDocs(sentence)
	for idx := range ti.docs {
		ti.sentenceScore = append(ti.sentenceScore, &SentenceScore{DocIndex: idx, Score: 0})
		for _, term := range sentence {
			ti.sentenceScore[idx].Score += ti.termTFMap[term][idx] * ti.termIDFMap[term]
		}
	}

	// 按 ti.sentenceScore[term] 从大到小排序
	sort.Slice(ti.sentenceScore, func(i, j int) bool {
		return ti.sentenceScore[i].Score > ti.sentenceScore[j].Score
	})
	return ti.docs[ti.sentenceScore[0].DocIndex].Words()
}

// 每一个词对于整个文档来说， tf idf 值是固定的，所以只需要计算一次
func (ti *TFIDF[T]) ScanDocs(sentence []string) {
	// 初始化 termTFMap（如果尚未初始化）
	if ti.termTFMap == nil {
		ti.termTFMap = make(map[string]map[int]float64)
	}
	docs := make([][]string, 0)
	for _, doc := range ti.docs {
		docs = append(docs, doc.Words())
	}

	for _, term := range sentence {
		// 检查 IDF 是否已缓存
		if _, ok := ti.termIDFMap[term]; !ok {
			// 计算并缓存 IDF
			idf := inverseDocumentFrequency(docs, term)
			ti.termIDFMap[term] = idf
		}

		// 检查当前文档的 TF 是否已缓存
		if _, ok := ti.termTFMap[term]; !ok {
			ti.termTFMap[term] = make(map[int]float64)
		}
		for i, doc := range ti.docs {
			if _, ok := ti.termTFMap[term][i]; !ok {
				tf := termFrequency(doc.Words(), term)
				ti.termTFMap[term][i] = tf
			}
		}
	}
}

func SentenceTFIDF(doc []string, docs [][]string, sentence []string) float64 {
	total := 0.0
	for _, term := range sentence {
		total += tfidf(doc, docs, term)
	}
	return total
}

// 计算单个词的 TF-IDF
func tfidf(doc []string, docs [][]string, term string) float64 {
	tf := termFrequency(doc, term)
	idf := inverseDocumentFrequency(docs, term)
	return tf * idf
}

func termFrequency(doc []string, term string) float64 {
	count := 0
	for _, word := range doc {
		if word == term {
			count++
		}
	}
	return float64(count) / float64(len(doc))
}

func inverseDocumentFrequency(docs [][]string, term string) (idf float64) {
	docCount := 0
	for _, doc := range docs {
		for _, word := range doc {
			if word == term {
				docCount++
				break
			}
		}
	}
	// 使用更通用的平滑公式：log((N+1)/(df+1)) + 1
	return math.Log(float64(len(docs)+1)/float64(docCount+1)) + 1
}

// coverRate 计算 doc 在 Docs 每个文件被覆盖的百分比
// 比如 123 在 12345 中被覆盖的百分比是 3/3=1.0
// 126 在 12345 中被覆盖的百分比是 2/3=0.6666666666666666
func QueryCoverage[T Tokenizable](query []string, docs []T) []float64 {
	rates := make([]float64, len(docs))
	for i := range docs {
		rates[i] = queryCoverage(query, docs[i].Words())
	}
	return rates
}

func queryCoverage(query []string, target []string) float64 {
	targetCount := make(map[string]int)
	for _, word := range target {
		targetCount[word]++
	}

	matched := 0
	for _, word := range query {
		if targetCount[word] > 0 {
			matched++
			targetCount[word]-- // 避免重复匹配
		}
	}

	return float64(matched) / float64(len(query))
}

func ZScoreStandardize(data []float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, errors.New("数据不能为空")
	}

	// 计算均值
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))

	// 计算标准差
	variance := 0.0
	for _, v := range data {
		variance += (v - mean) * (v - mean)
	}
	stdDev := math.Sqrt(variance / float64(len(data)))

	if stdDev == 0 {
		return nil, errors.New("标准差为0，无法标准化")
	}

	standardized := make([]float64, len(data))
	for i, v := range data {
		standardized[i] = (v - mean) / stdDev
	}
	return standardized, nil
}
