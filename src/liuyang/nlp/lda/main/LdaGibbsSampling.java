package liuyang.nlp.lda.main;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.conf.ConstantConfig;
import liuyang.nlp.lda.conf.PathConfig;

/**Liu Yang's implementation of Gibbs Sampling of LDA
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */

public class LdaGibbsSampling {
	
	public static class modelparameters {
		float alpha = 0.5f; //usual value is 50 / K
		float beta = 0.1f;//usual value is 0.1
		int topicNum = 100;
		int iteration = 100;
		int saveStep = 10;
		int beginSaveIters = 50;
	}
	
	/**Get parameters from configuring file. If the 
	 * configuring file has value in it, use the value.
	 * Else the default value in program will be used
	 * @param ldaparameters
	 * @param parameterFile
	 * @return void
	 */
	private static void getParametersFromFile(modelparameters ldaparameters,
			String parameterFile) {
		// TODO Auto-generated method stub
		ArrayList<String> paramLines = new ArrayList<String>();
		FileUtil.readLines(parameterFile, paramLines);
		for(String line : paramLines){
			String[] lineParts = line.split("\t");
			switch(parameters.valueOf(lineParts[0])){
			case alpha:
				ldaparameters.alpha = Float.valueOf(lineParts[1]);
				break;
			case beta:
				ldaparameters.beta = Float.valueOf(lineParts[1]);
				break;
			case topicNum:
				ldaparameters.topicNum = Integer.valueOf(lineParts[1]);
				break;
			case iteration:
				ldaparameters.iteration = Integer.valueOf(lineParts[1]);
				break;
			case saveStep:
				ldaparameters.saveStep = Integer.valueOf(lineParts[1]);
				break;
			case beginSaveIters:
				ldaparameters.beginSaveIters = Integer.valueOf(lineParts[1]);
				break;
			}
		}
	}
	
	public enum parameters{
		alpha, beta, topicNum, iteration, saveStep, beginSaveIters;
	}
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		//文档路径
		String originalDocsPath = PathConfig.ldaDocsPath;
		//结果集
		String resultPath = PathConfig.LdaResultsPath;
		//Lda参数
		String parameterFile= ConstantConfig.LDAPARAMETERFILE;

		//模型参数,如果文件中没有设定,那么就选择程序中设定的参数
		modelparameters ldaparameters = new modelparameters();


		getParametersFromFile(ldaparameters, parameterFile);

		//文档集合
		Documents docSet = new Documents();
		docSet.readDocs(originalDocsPath);//文档路径

		System.out.println("wordMap size " + docSet.termToIndexMap.size());

		FileUtil.mkdir(new File(resultPath));//产生 文件

		LdaModel model = new LdaModel(ldaparameters); //参数
		System.out.println("1 Initialize the model ...");
		model.initializeModel(docSet);//初始化 模型

		System.out.println("2 Learning and Saving the model ...");
		model.inferenceModel(docSet);//推理 模型

		System.out.println("3 Output the final model ...");
		model.saveIteratedModel(ldaparameters.iteration, docSet);//迭代次数，

		System.out.println("Done!");
	}
}
