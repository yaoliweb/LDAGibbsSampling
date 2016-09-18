package liuyang.nlp.lda.main;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.com.Stopwords;

/**Class for corpus which consists of M documents
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */

public class Documents {
	
	ArrayList<Document> docs; 
	Map<String, Integer> termToIndexMap;
	ArrayList<String> indexToTermMap;
	Map<String,Integer> termCountMap;
	
	public Documents(){
		docs = new ArrayList<Document>();
		termToIndexMap = new HashMap<String, Integer>();//term 到 index Map ；词语到索引
		indexToTermMap = new ArrayList<String>();//index 到 Term Map；索引到Term List
		termCountMap = new HashMap<String, Integer>();//term 数量；词语 --> 数量
	}
	
	public void readDocs(String docsPath){
		for(File docFile : new File(docsPath).listFiles()){ // 遍历读取 文档 形成集合
			Document doc = new Document(docFile.getAbsolutePath(), termToIndexMap, indexToTermMap, termCountMap);
			docs.add(doc);
		}
	}

	// 静态内部类
	public static class Document {	
		private String docName; //文档名称
		int[] docWords;	//文档单词

		// 内部构造函数
		public Document(String docName, Map<String, Integer> termToIndexMap, ArrayList<String> indexToTermMap, Map<String, Integer> termCountMap){
			this.docName = docName;//文档名称
			//Read file and initialize word index array
			ArrayList<String> docLines = new ArrayList<String>();//文档行数
			//一篇文章中的所有单词
			ArrayList<String> words = new ArrayList<String>();//文档所有单词
			FileUtil.readLines(docName, docLines);//根据docName 将文件的每行读入到docLines

			for(String line : docLines){
				FileUtil.tokenizeAndLowerCase(line, words);//将每行数据分割 到单词里面
			}
			//去掉停止词和噪声词
			for(int i = 0; i < words.size(); i++){
				if(Stopwords.isStopword(words.get(i)) || isNoiseWord(words.get(i))){
					words.remove(i);
					i--;
				}
			}
			//Transfer word to index，将词 转换成 索引
			this.docWords = new int[words.size()];//docWords长度是所有单词的数量，
			// 存放的是 每个单词 对应的索引，会有多个单词对应同一个索引
			for(int i = 0; i < words.size(); i++){ //
				String word = words.get(i);
				if(!termToIndexMap.containsKey(word)){ // termToIndexMap:术语和Index 索引
					int newIndex = termToIndexMap.size();
					termToIndexMap.put(word, newIndex);// 单词 对应 索引
					indexToTermMap.add(word);// 索引对应着 单词， 1 表示 某个单词
					termCountMap.put(word, new Integer(1));// 单词 计数
					docWords[i] = newIndex; //记下单词的索引
				} else {
					docWords[i] = termToIndexMap.get(word);//根据单词取得索引，将该索引放入 docWord中。存放每个单词的索引
					termCountMap.put(word, termCountMap.get(word) + 1);
				}
			}
			words.clear();
		}

		//是不是噪声单词
		public boolean isNoiseWord(String string) {
			// TODO Auto-generated method stub
			string = string.toLowerCase().trim();
			Pattern MY_PATTERN = Pattern.compile(".*[a-zA-Z]+.*");
			Matcher m = MY_PATTERN.matcher(string);
			// filter @xxx and URL
			if(string.matches(".*www\\..*") || string.matches(".*\\.com.*") || 
					string.matches(".*http:.*") )
				return true;
			if (!m.matches()) {
				return true;
			} else
				return false;
		}
		
	}
}
