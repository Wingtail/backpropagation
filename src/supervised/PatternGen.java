package supervised;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
public class PatternGen {
	int[] operation;
	int startNum;
	int difficulty = 3;
	int[] stepNum = new int[6];
	public int[] targetar;
	int patterns = 176;
	int vpatterns = 24;
	int length = 10;
	int fudge = 100;
	File file = new File("Pattern.txt");
	File file1 = new File("Problem.txt");
	FileWriter writer;
	FileWriter writer1;
	PatternGen(){
		
	}
	public static void main(String[] args)
	{
		int i;
		int j;
		int k;
		PatternGen pat = new PatternGen();
		Random rand = new Random();
		for(i=0;i<pat.stepNum.length;i++)
		{
			pat.stepNum[i] = rand.nextInt(10);
		}
		pat.startNum = rand.nextInt(10);
		pat.operation = new int[pat.difficulty];
		pat.targetar = new int[pat.patterns+pat.vpatterns];
		for(i=0;i<pat.difficulty;i++)
		{
			pat.operation[i] = rand.nextInt(2);
		}
		try {
			double num = 0;
	    	  pat.file.createNewFile();
			    pat.writer = new FileWriter(pat.file); 
			   pat.file1.createNewFile();
			   pat.writer1 = new FileWriter(pat.file1);
			   for(i=0;i<pat.patterns;i++)
			   {
				   pat.writer.flush();
				   pat.startNum = rand.nextInt(100);
				   //pat.writer.write("{");
				   for(j=0;j<pat.length;j++)
				   {
					   
					   if(j==0)
					   {
						   num = pat.startNum;
						   pat.writer.write(num/pat.fudge+"\n");
					   }else{
						   if(pat.operation[j%3] == 0)
						   {
							   num += pat.stepNum[j%3];
						   }else if(pat.operation[j%3] == 1)
						   {
							   num -= pat.stepNum[j%3];
						   }else if(pat.operation[j%3] == 2)
						   {
							   num *= pat.stepNum[j%3];
						   }
						   if(j+1 == pat.length)
						   {
							   pat.targetar[i] = (int) num;
							   break;
						   }if(j+2 == pat.length)
						   {
							   String p = "" + num/pat.fudge;
							   pat.writer.write(p);
						   }else{
						   pat.writer.write(num/pat.fudge+"\n" );
						   }
					   }
				   }
				   pat.writer.write("\n},\n");
			   }
			   
			   //pat.writer.write("");
			   pat.writer.write("Vali\n");
			   for(i=pat.patterns;i<pat.patterns+pat.vpatterns;i++)
			   {
				   pat.writer.flush();
				   pat.startNum = rand.nextInt(100);
				   
				   for(j=0;j<pat.length;j++)
				   {
					   if(i+1 == pat.patterns+pat.vpatterns && j==0)
					   {
						   pat.writer.write("Test\n" );
						   if(j+1 == pat.length)
						   {
							   //System.out.println("...?");
							   break;
						   }
					   }
					   if(j==0)
					   {
						   num = pat.startNum;
						   pat.writer.write(num/pat.fudge+"\n");
						   if(i+1 == pat.patterns+pat.vpatterns)
						   {
							   pat.writer1.write(num/pat.fudge+"\n");
							   System.out.print(num/pat.fudge+", ");
						   }
					   }else{
						   if(pat.operation[j%3] == 0)
						   {
							   num += pat.stepNum[j%3];
						   }else if(pat.operation[j%3] == 1)
						   {
							   num -= pat.stepNum[j%3];
						   }else if(pat.operation[j%3] == 2)
						   {
							   num *= pat.stepNum[j%3];
						   }
						   if(j+1 == pat.length)
						   {
							   pat.targetar[i] = (int) num;
							   break;
						   }if(j+2 == pat.length)
						   {
							   String p = "" + num/pat.fudge;
							   pat.writer.write(p);
							   if(i+1 == pat.patterns+pat.vpatterns)
							   {
								   pat.writer1.write(p);
								   System.out.print(num/pat.fudge+", ");
							   }
						   }else{
							   if(i+1==pat.patterns+pat.vpatterns)
							   {
								   pat.writer1.write(num/pat.fudge+"\n");
								   System.out.print(num/pat.fudge+", ");
							   }
						   pat.writer.write(num/pat.fudge+"\n");
						   }
					   }
				   }
				   pat.writer.write("\n},\n");
			   }
			   
			  // pat.writer.write("\n");
			   pat.writer.write("targetT\n"); 
			   for(i=0;i<pat.patterns;i++)
			   {
				   String st = "" + (double)pat.targetar[i]/pat.fudge;
				  pat.writer.write(st+"\n");
			   }
			   pat.writer.write("},\n"); 
			   
			   //pat.writer.write("\n");
			   pat.writer.write("targetV\n"); 
			   for(i=pat.patterns;i<pat.patterns+pat.vpatterns;i++)
			   {
				   String st = "" + (double)pat.targetar[i]/pat.fudge;
				  pat.writer.write(st+"\n");
			   }
			   pat.writer.write("},\n"); 
			  
				 pat.writer.close();
				 pat.writer1.close();
	      // creates a FileWriter Object

	      // Writes the content to the file
			//Train();	
		  
	      }catch(IOException e){
	    	  
	      }
		
	}
	
}
