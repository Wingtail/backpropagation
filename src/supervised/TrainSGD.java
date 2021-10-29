package supervised;
import java.util.*;
public class TrainSGD {
	public double[] gBatch = {0.0};
	
	ArrayList<ArrayList<ArrayList<Double>>> gradientGroup = new ArrayList<ArrayList<ArrayList<Double>>>();
	ArrayList<Double> Node = new ArrayList<Double>();
	double gradients;
	ArrayList<ArrayList<Double>> Layer = new ArrayList<ArrayList<Double>>();
	
	ArrayList<Double> inputLayer = new ArrayList<Double>();
	ArrayList<ArrayList<Node>> hiddenLayer = new ArrayList<ArrayList<Node>>();
	ArrayList<Node> outputLayer = new ArrayList<Node>();
	double[][] weights;
	double learnRate;
	double momentum;
	double error;
	
	double autoCorr = 0.95;
	double masterStep = 0.001;
	double fudge = 0.0000001;
	double[] historicalgrad = {0};
	
	double[][] nodedelta = null;
	double[] deltaval = {0};
	int i;
	int j;
	int k;
	int num = 0;
	boolean flag = false;
	
	public TrainSGD() //Connect Backpropagation class correctly!! hiddenLayer = null error (Solved)
	{
		
	}
	public void GethistoryGrad(double[] history)
	{
		historicalgrad = history;
	}
	public void Getgrad(double[] js)
	{
		gBatch = js;
	}
	public void Getdval(double [] kj)
	{
		deltaval = kj;
	}
	public void Train(int c)
	{
		nodeDelta();
		gDescent();
		gradientSummate();
	
		if(c==1)
		{
			weightUp();
			gBatch = new double[0];
		}
	}
	public void setLayers(ArrayList<Double> inLayer, ArrayList<ArrayList<Node>> hidLayer, ArrayList<Node> outLayer)
	{
		inputLayer = inLayer;
		hiddenLayer = hidLayer;
		outputLayer = outLayer;
	}
	public void setWeights(double[][] weight)
	{
		weights = new double[weight.length][0];
		for(i=0;i<weight.length;i++)
		{
			weights[i] = new double[weight[i].length];
		}
		this.weights = weight;
		
	}
	public void setvalues(double e, double lR, double a)
	{
		error = e;
		learnRate = lR;
		momentum = a;
	}
	public double dsigmoid(double val) //Derivative sigmoid
	{
		val = (1/Math.cosh(val))*(1/Math.cosh(val));
		return val;
	}
	public void nodeDelta()
	{
		
		nodedelta = new double[hiddenLayer.size()+1][0];
			nodedelta[nodedelta.length-1] = new double[outputLayer.size()];
			for(i=0;i<outputLayer.size();i++)
			{
				nodedelta[nodedelta.length-1][i] = -error*dsigmoid(outputLayer.get(i).sum);
			}
//Good
		for(i=nodedelta.length-2;i>=0;i--)
		{
				nodedelta[i] = new double[hiddenLayer.get(i).size()];
				for(j=0;j<nodedelta[i].length;j++)
				{
					double imsiS = 0;
					for(k=0;k<nodedelta[i+1].length;k++) //Have to take a look at this, need debugging.
					{
						imsiS += (nodedelta[i+1][k]*weights[i+1][(j*nodedelta[i+1].length)+k]); 
					}
					nodedelta[i][j] = dsigmoid(hiddenLayer.get(i).get(j).sum) * imsiS;
				}
		}
	//}
	}
	
	public void gDescent()
	{	
		gradientGroup = new ArrayList<ArrayList<ArrayList<Double>>>();
		Layer = new ArrayList<ArrayList<Double>>(); //ArrayList<ArrayList<Double>>() would be also ArrayList<Node>()
		for(i=0;i<=inputLayer.size();i++)
		{
			Node = new ArrayList<Double>();
			for(j=0;j<hiddenLayer.get(0).size();j++)
			{
					if(i==inputLayer.size())
					{
						gradients = 1 * nodedelta[0][j];
					}else{
						gradients = inputLayer.get(i) * nodedelta[0][j]; 
					}
					Node.add(gradients);
			}
			Layer.add(Node);
		}
		gradientGroup.add(Layer);
		
		for(i=0;i<(hiddenLayer.size()-1);i++)
		{
			Layer = new ArrayList<ArrayList<Double>>();
			for(j=0;j<=hiddenLayer.get(i).size();j++)
			{
				Node = new ArrayList<Double>();
				for(k=0;k<hiddenLayer.get(i+1).size();k++)
				{
					if(j==hiddenLayer.get(i).size())
					{
						gradients = 1 * nodedelta[i+1][k]; //Suspicion error
					}else{
						gradients = hiddenLayer.get(i).get(j).eval * nodedelta[i+1][k]; //Still in error...
					}
					Node.add(gradients);
				}
				Layer.add(Node);
			}
			gradientGroup.add(Layer);
		}
		Layer = new ArrayList<ArrayList<Double>>();
		for(i=0;i<=hiddenLayer.get(hiddenLayer.size()-1).size();i++)
		{
			Node = new ArrayList<Double>();
			for(j=0;j<outputLayer.size();j++)
			{
					if(i==hiddenLayer.get(hiddenLayer.size()-1).size())
					{
						gradients = 1 * nodedelta[nodedelta.length-1][j];
					}else{
						gradients = hiddenLayer.get(hiddenLayer.size()-1).get(i).eval * nodedelta[nodedelta.length-1][j];
					}
					Node.add(gradients);
			}
			Layer.add(Node);
			//Work on this, the last hiddenLayer to the output Layer.
		}
		gradientGroup.add(Layer);
	}
	
	public void gradientSummate()
	{
		num = 0;
		if(hiddenLayer.size() > 1)
		{
			for(i=1;i<hiddenLayer.size();i++)
			{
				num += (hiddenLayer.get(i).size())*(hiddenLayer.get(i-1).size()+1); //+1 is because of bias value.
			}
			num += ((hiddenLayer.get(hiddenLayer.size()-1).size()+1) * outputLayer.size()); // +1 is because of bias value
			num += (inputLayer.size()+1) * hiddenLayer.get(0).size(); //+1 is because of bias
		}else if(hiddenLayer.size()==1){
			num += (hiddenLayer.get(0).size()+1) * (outputLayer.size());
			num += (inputLayer.size()+1) * (hiddenLayer.get(0).size());
		}else{
			//Elaborate for extension of the program.
		}
		//System.out.println("Number of Connections: "+num); //Representing number of connections
	//	if(num != gBatch.length)
		//{
			gBatch = new double[num];
			if(historicalgrad.length <= 1 && historicalgrad[0] == 0)
			{
				historicalgrad = new double[num];
			}
	//	}
		int o = 0;
		for(i=0;i<gradientGroup.size();i++) //Out of all layers
		{
			for(j=0;j<gradientGroup.get(i).size();j++) //Out of all nodes
			{
				for(k=0;k<gradientGroup.get(i).get(j).size();k++) //Out of all gradients
				{
					gBatch[o] = gradientGroup.get(i).get(j).get(k); //201 marker
					if(historicalgrad[0] == 0)
					{
						historicalgrad[o] = (gBatch[o]*gBatch[o]);
					}else{
						historicalgrad[o] = (autoCorr*historicalgrad[o] + ((1-autoCorr)*(gBatch[o]*gBatch[o])));
					}
					gBatch[o] = (gBatch[o] / (fudge+Math.sqrt(historicalgrad[o])));
					o++;
				}
			}
		}
		
	}
	
	public void weightUp()
	{
		int o=0;
		for(i=0;i<weights.length;i++)
		{
			for(j=0;j<weights[i].length;j++)
			{
				double deltaW = weights[i][j];
				//weights[i][j] += (0.1*((gBatch[o]*learnRate)+(momentum*deltaval[j])));
				weights[i][j] += (masterStep*gBatch[o])+(momentum*deltaval[j]);
				double imsi = weights[i][j] - deltaW;
				deltaval[j] = imsi;
				o++;
			}
		}
	}
}
