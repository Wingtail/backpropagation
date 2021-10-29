package supervised;
import java.util.*;
public class MainEval {
	
	public int[][] input = {{1,1,1,1},{0,0,0,1},{0,0,1,0}}; //The input data matrix
	public int[] target = {0,1,0};							// Target value relative to input data matrix
	public double[][] weights = new double[0][0];			//Weight Matrix(2x2)
	public double learnRate = 0.75;							//The Learning rate of the neural network.
	
	public int inputNodes = 4;
	public int[][] hiddenNodes = {{1}}; //No hiddenNodes = {{1}}
	public int outputNodes = 1;
	public int phase = 0;
	public double netError = 0;
	public double error = 0;
	
	public int bias = 1;
	public int trainPercent = 70;
	
	ArrayList<Double> recordErrors = new ArrayList<Double>();
	ArrayList<Double> inputLayer = new ArrayList<Double>();
	ArrayList<ArrayList<Node>> hiddenLayer = new ArrayList<ArrayList<Node>>();
	ArrayList<Node> outputLayer = new ArrayList<Node>();
	
	double[] output;

	public static void main(String[] args)
	{
		MainEval main = new MainEval();
	}
	MainEval(){
		do{
			while(phase<input.length)
			{
				
				setWeights();
				initInput();
				TrainNodes();
				removeInput();
				phase++;
				System.out.println("Output"+output[0]);

				for(int i=0;i<weights.length;i++)
				{
					for(int j=0;j<weights[i].length;j++)
					{
						System.out.println("Weights: "+weights[i][j]);
					}
				}
			}
			CalcNetError();
			System.out.println("NetError"+netError);
			phase = 0;
		}while(netError!=0.00);
		
		System.out.println("Training Complete!");
		System.out.println("Exporting weight values...");
		for(int i=0;i<weights.length;i++)
		{
			for(int j=0;j<weights[i].length;j++)
			{
				System.out.println(weights[i][j]);
			}
		}
		
		/*System.out.println("Testing...");
		int[][] inputss = {{0,1,0,0},{0,1,0,1},{0,1,1,0},{0,1,1,1},{0,0,1,0},{0,0,1,1}};
		for(int i=0;i<input.length;i++)
		{
			for(int j=0;j<input[i].length;j++)
			{
				input[i][j] = inputss[i][j];
			}
		}

		while(phase<input.length)
		{
			setWeights();
			initInput();
			TestNodes();
			removeInput();
			phase++;
			System.out.println("Output"+output[0]);

			for(int i=0;i<weights.length;i++)
			{
				for(int j=0;j<weights[i].length;j++)
				{
					System.out.println("Weights: "+weights[i][j]);
				}
			}
		}
		CalcNetError();
		System.out.println("NetError"+netError);*/
		
	}
	public int ratioPercent(int input, int percent)
	{
		int percent1 = (int)(Math.round(input*input*(percent/100)*100.0)/100.0);
		return percent1;
	}
	public void setWeights()
	{
		int i;
		int j;
		int k;
		if(weights.length <= 0)
		{
		if(hiddenNodes[0][0]==1)
		{
			weights = new double[(1+0+1)-1][0];
			for(i=0;i<weights.length;i++)
			{
					weights[i] = new double[inputNodes*outputNodes];
			}
		}else{
			weights = new double[(1+hiddenNodes.length+1)-1][0];
			for(i=0;i<weights.length;i++)
			{
				if(i==weights.length-1){
					weights[i] = new double[hiddenNodes[i-1].length*outputNodes];
				}else{
					if(i-1>0)
					{
						weights[i] = new double[hiddenNodes[i-1].length*hiddenNodes[i].length];
					}else{
						weights[i] = new double[inputNodes*hiddenNodes[i].length];
					}
				}
			}
			
		}
			for(i=0;i<weights.length;i++){
				for(j=0;j<weights[i].length;j++)
				{
					weights[i][j] = 0.0;
				}
			}
	
			Random rand = new Random();
			double high = 1.0;
			double low = -1.0;
			for(i=0;i<weights.length;i++)
			{
				for(j=0;j<weights[i].length;j++)
				{
						weights[i][j] = high+(rand.nextDouble()*(low-high));
				}
			}
		}
	}
	public void initInput()
	{
		int k;
		int i;
		int j;
		
			for(j=0;j<input[phase].length;j++)
				inputLayer.add((double) input[phase][j]);
			System.out.println("Contents of input: "+inputLayer);
	}
	public void removeInput()
	{
		int i;
		for(i=0;i<input[0].length;i++)
		{
			inputLayer.remove(0);
		}
	}
	public void TrainNodes()
	{
		setNodes();
		passOutput();
		calcNewWeight();
		flushOutLayer();
	}
	public void TestNodes()
	{
		setNodes();
		passOutput();
		flushOutLayer();
	}
	public void setNodes(){
		int i;
		int j;
		int k;
		if(hiddenNodes[0][0]==1)
		{
			Node node = setNodes1(0);
			outputLayer.add(node);
		}else{
			for(i=0;i<hiddenNodes.length;i++)
			{
				ArrayList<Node> temp = new ArrayList<Node>();
				for(j=0;j<hiddenNodes[i][0];j++)
				{
					/*Node node = what node..;
					temp.add(node);*/
					Node node = setNodes1(i);
					temp.add(node);
				}
				hiddenLayer.add(temp);
			}
			for(i=0;i<outputNodes;i++)
			{
				Node node = setNodes1(hiddenLayer.size());
				outputLayer.add(node);
			}
		}
	}
	public Node setNodes1(int curLayer){
		int i;
		int j;
		int k;
		double[] tempIN = null;
		double[] tempW = null;
		Node node = null;
		//pre-set weights and output values 
		if(curLayer==0)
		{
			tempIN = new double[inputLayer.size()];
			tempW = new double[weights[curLayer].length];
			//double[] tempW = new double[weights[0].length];
			for(i=0;i<inputLayer.size();i++)
			{
				tempIN[i] = inputLayer.get(i);
				tempW[i] = weights[curLayer][i*hiddenNodes[curLayer].length];
			}
			node = new Node(tempIN, tempW);
		}else{
			tempIN = new double[hiddenLayer.get(curLayer/*-1*/).size()]; // We have to figure out why I put curLayer-1... Right now, it does not make sense.
			tempW = new double[weights[curLayer/*-1*/].length];
			for(i=0;i<hiddenLayer.get(curLayer/*-1*/).size();i++)
			{
				tempIN[i] = (double)hiddenLayer.get(curLayer/*-1*/).get(i).eval;
				tempW[i] = weights[curLayer/*-1*/][i*hiddenNodes[curLayer].length];
			}
			node = new Node(tempIN, tempW);
			
		}
		return node;
	}
	public void passOutput()
	{
		int i;
		int j;
		output = new double[outputLayer.size()];
		for(i=0;i<output.length;i++)
		{
			output[i] = outputLayer.get(i).eval;
		}
	}
	public void flushOutLayer()
	{
		int i;
		int size = outputLayer.size();
		for(i=0;i<size;i++)
		{
			outputLayer.remove(0);
		}
		size = hiddenLayer.size();
		for(i=0;i<size;i++)
		{
			hiddenLayer.remove(0);
		}
	}
	public double CalcError()
	{
		double error = target[phase] - output[0];
		this.error = error;
		return error;
	}
	public void CalcNetError()
	{
		int i;
		int j;
		double net = 0;
		for(i=0;i<recordErrors.size();i++)
		{
			net += ((recordErrors.get(i)*recordErrors.get(i)));
		}
		//netError = Math.round(net*100.0)/100.0;
		netError = net;
		int size = recordErrors.size();
		for(i=0;i<size;i++)
		{
			recordErrors.remove(0);
		}
		
	}
	public void roundWeights()
	{
		int i;
		int j;
		for(i=0;i<weights.length;i++)
		{
			for(j=0;j<weights[i].length;j++)
			{
				weights[i][j] = (double)Math.round(weights[i][j]*100.000)/100.000;
			}
		}
	}
	public void calcNewWeight()
	{
		int i;
		int j;
		for(i=0;i<weights.length;i++)
		{
			for(j=0;j<weights[i].length;j++)
			{
				double error = CalcError();
				System.out.println("Error"+error);
				recordErrors.add(error);
				double weightVal = learnRate*inputLayer.get((int)(Math.round(j/outputLayer.size()*100.0)/100.0))*error;
				weights[i][j] = weightVal+weights[i][j];
			}
		}	
		roundWeights();
	}
	
}
