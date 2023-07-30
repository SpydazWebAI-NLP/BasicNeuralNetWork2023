Imports System.IO
Imports System.Web.Script.Serialization
Imports System.Windows.Forms
Imports System.Xml.Serialization
Imports SpydazWebAI.NeuralNetWork.Basic_NLP.SingleLayerNeuralNetwork

Namespace NeuralNetworkFactory


    Public MustInherit Class NeuralNetworkFactory
        Public Enum NetworkType
            FeedForwards
            BackPropergation
            None
        End Enum

        ''' <summary>
        ''' Each layer consists of neurons(nodes) the training cases also use an input layer and an
        ''' output layer
        ''' </summary>
        ''' <remarks></remarks>
        Public Class Layer
            ''' <summary>
            ''' Each layer consists of nodes (neurons) these are each individual. all layers contain
            ''' nodes, Used for neural network inputs / outputs / hidden nodes
            ''' </summary>
            ''' <remarks></remarks>
            Public Class Neuron

                ''' <summary>
                ''' The input of the node is the collective sum of the inputs and their respective weights
                ''' </summary>
                ''' <remarks></remarks>
                Public input As Double

                ''' <summary>
                ''' the output of the node is also relational to the transfer function used
                ''' </summary>
                ''' <remarks></remarks>
                Public output As Double

                ''' <summary>
                ''' There is a value attached with dendrite called weight. The weight associated with a
                ''' dendrites basically determines the importance of incoming value. A weight with
                ''' larger value determines that the value from that particular neuron is of higher
                ''' significance. To achieve this what we do is multiply the incoming value with weight.
                ''' So no matter how high the value is, if the weight is low the multiplication yields
                ''' the final low value.
                ''' </summary>
                ''' <remarks></remarks>
                Public weight As Double

                ''' <summary>
                ''' Add biasing to the Perceptron
                ''' </summary>
                Public bias As Double

                ''' <summary>
                ''' Constructor Single Input
                ''' </summary>
                Public Sub New()
                    CreateRandWeight(0, 1)
                    bias = 0.1
                End Sub

                ''' <summary>
                ''' Initial Weights can be determined by the number of hidden nodes and the number of
                ''' input nodes this is a rule of thumb
                ''' </summary>
                ''' <remarks></remarks>
                Public Sub CreateRandWeight(ByRef InputLow As Integer, ByRef InputHigh As Integer)
                    Randomize()
                    Dim value As Integer = CInt(Int((InputHigh * Rnd()) + InputLow))
                    weight = value
                End Sub

                ''' <summary>
                ''' Sets input of the node
                ''' </summary>
                ''' <param name="value"></param>
                Public Sub SetInput(ByRef value As Double)
                    input = value
                End Sub

                ''' <summary>
                ''' Activates Node and sets the output for the node
                ''' </summary>
                ''' <param name="Activation">Activation Function</param>
                ''' <remarks>ActivationFunction(Node.input * Node.weight)</remarks>
                Public Sub ActivateNode(ByRef Activation As TransferFunctionType)
                    output = TransferFunction.EvaluateTransferFunct(Activation, NodeTotal())
                End Sub

                ''' <summary>
                ''' Produces a node total which can be fed to the activation function (Stage 1)
                ''' (input * weight)
                ''' </summary>
                ''' <returns>Node input * Node Weight</returns>
                ''' <remarks></remarks>
                Public Function NodeTotal() As Double
                    Return input * weight + bias
                End Function

                ''' <summary>
                ''' Recalcualtes the weight for this node
                ''' </summary>
                ''' <param name="ActualOutput">Output of node</param>
                ''' <param name="ExpectedOutput">Expected Output of node</param>
                Public Sub RecalculateWeight(ByRef ActualOutput As Double, ByRef ExpectedOutput As Double)
                    Dim NodeError As Double = ActualOutput - ExpectedOutput
                    Dim Delta = NodeError * TransferFunction.EvaluateTransferFunctionDerivative(TransferFunctionType.Sigmoid, ActualOutput)
                    weight += Delta
                    bias += Delta
                End Sub

                ''' <summary>
                ''' Returns output from Node Given Activation Function
                ''' </summary>
                ''' <param name="Activation"></param>
                ''' <returns></returns>
                Public Function Compute(ByRef Activation As TransferFunctionType) As Double
                    Return TransferFunction.EvaluateTransferFunct(Activation, NodeTotal())
                End Function

                ''' <summary>
                ''' deserialize object from Json
                ''' </summary>
                ''' <param name="Str">json</param>
                ''' <returns></returns>
                Public Shared Function FromJson(ByRef Str As String) As Neuron
                    Try
                        Dim Converter As New JavaScriptSerializer
                        Dim diag As Neuron = Converter.Deserialize(Of Neuron)(Str)
                        Return diag
                    Catch ex As Exception
                        Dim Buttons As MessageBoxButtons = MessageBoxButtons.OK
                        MessageBox.Show(ex.Message, "ERROR", Buttons)
                    End Try
                    Return Nothing
                End Function

                ''' <summary>
                ''' Serializes object to json
                ''' </summary>
                ''' <returns> </returns>
                Public Function ToJson() As String
                    Dim Converter As New JavaScriptSerializer
                    Return Converter.Serialize(Me)
                End Function

                ''' <summary>
                ''' Transfer Function used in the calculation of the following layer
                ''' </summary>
                Public Structure TransferFunction
                    ''' <summary>
                    ''' These are the options of transfer functions available to the network
                    ''' This is used to select which function to be used:
                    ''' The derivative function can also be selected using this as a marker
                    ''' </summary>
                    Public Enum TransferFunctionType
                        none
                        sigmoid
                        HyperbolTangent
                        BinaryThreshold
                        RectifiedLinear
                        Logistic
                        StochasticBinary
                        Gaussian
                        Signum
                    End Enum
                    ''' <summary>
                    ''' Returns a result from the transfer function indicated ; Non Derivative
                    ''' </summary>
                    ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
                    ''' <param name="Input">Input value for node/Neuron</param>
                    ''' <returns>result</returns>
                    Public Shared Function EvaluateTransferFunct(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
                        EvaluateTransferFunct = 0
                        Select Case TransferFunct
                            Case TransferFunctionType.none
                                Return Input
                            Case TransferFunctionType.sigmoid
                                Return Sigmoid(Input)
                            Case TransferFunctionType.HyperbolTangent
                                Return HyperbolicTangent(Input)
                            Case TransferFunctionType.BinaryThreshold
                                Return BinaryThreshold(Input)
                            Case TransferFunctionType.RectifiedLinear
                                Return RectifiedLinear(Input)
                            Case TransferFunctionType.Logistic
                                Return Logistic(Input)
                            Case TransferFunctionType.Gaussian
                                Return Gaussian(Input)
                            Case TransferFunctionType.Signum
                                Return Signum(Input)
                        End Select
                    End Function

                    ''' <summary>
                    ''' Returns a result from the transfer function indicated ; Non Derivative
                    ''' </summary>
                    ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
                    ''' <param name="Input">Input value for node/Neuron</param>
                    ''' <returns>result</returns>
                    Public Shared Function EvaluateTransferFunctionDerivative(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
                        EvaluateTransferFunctionDerivative = 0
                        Select Case TransferFunct
                            Case TransferFunctionType.none
                                Return Input
                            Case TransferFunctionType.sigmoid
                                Return SigmoidDerivitive(Input)
                            Case TransferFunctionType.HyperbolTangent
                                Return HyperbolicTangentDerivative(Input)
                            Case TransferFunctionType.Logistic
                                Return LogisticDerivative(Input)
                            Case TransferFunctionType.Gaussian
                                Return GaussianDerivative(Input)
                        End Select
                    End Function

                    ''' <summary>
                    ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
                    ''' binary data.
                    ''' </summary>
                    ''' <param name="Value"></param>
                    ''' <returns></returns>
                    ''' <remarks></remarks>
                    Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

                        ' Z = Bias+ (Input*Weight)
                        'TransferFunction
                        'If Z > 0 then Y = 1
                        'If Z < 0 then y = 0

                        Return If(Value < 0 = True, 0, 1)
                    End Function

                    Private Shared Function Gaussian(ByRef x As Double) As Double
                        Gaussian = Math.Exp((-x * -x) / 2)
                    End Function

                    Private Shared Function GaussianDerivative(ByRef x As Double) As Double
                        GaussianDerivative = Gaussian(x) * (-x / (-x * -x))
                    End Function

                    Private Shared Function HyperbolicTangent(ByRef Value As Double) As Double
                        ' TanH(x) = (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))

                        Return Math.Tanh(Value)
                    End Function

                    Private Shared Function HyperbolicTangentDerivative(ByRef Value As Double) As Double
                        HyperbolicTangentDerivative = 1 - (HyperbolicTangent(Value) * HyperbolicTangent(Value)) * Value
                    End Function

                    'Linear Neurons
                    ''' <summary>
                    ''' in a liner neuron the weight(s) represent unknown values to be determined the
                    ''' outputs could represent the known values of a meal and the inputs the items in the
                    ''' meal and the weights the prices of the individual items There are no hidden layers
                    ''' </summary>
                    ''' <remarks>
                    ''' answers are determined by determining the weights of the linear neurons the delta
                    ''' rule is used as the learning rule: Weight = Learning rate * Input * LocalError of neuron
                    ''' </remarks>
                    Private Shared Function Linear(ByRef value As Double) As Double
                        ' Output = Bias + (Input*Weight)
                        Return value
                    End Function

                    'Non Linear neurons
                    Private Shared Function Logistic(ByRef Value As Double) As Double
                        'z = bias + (sum of all inputs ) * (input*weight)
                        'output = Sigmoid(z)
                        'derivative input = z/weight
                        'derivative Weight = z/input
                        'Derivative output = output*(1-Output)
                        'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

                        Return 1 / 1 + Math.Exp(-Value)
                    End Function

                    Private Shared Function LogisticDerivative(ByRef Value As Double) As Double
                        'z = bias + (sum of all inputs ) * (input*weight)
                        'output = Sigmoid(z)
                        'derivative input = z/weight
                        'derivative Weight = z/input
                        'Derivative output = output*(1-Output)
                        'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

                        Return Logistic(Value) * (1 - Logistic(Value))
                    End Function

                    Private Shared Function RectifiedLinear(ByRef Value As Double) As Double
                        'z = B + (input*Weight)
                        'If Z > 0 then output = z
                        'If Z < 0 then output = 0
                        If Value < 0 = True Then

                            Return 0
                        Else
                            Return Value
                        End If
                    End Function

                    ''' <summary>
                    ''' the log-sigmoid function constrains results to the range (0,1), the function is
                    ''' sometimes said to be a squashing function in neural network literature. It is the
                    ''' non-linear characteristics of the log-sigmoid function (and other similar activation
                    ''' functions) that allow neural networks to model complex data.
                    ''' </summary>
                    ''' <param name="Value"></param>
                    ''' <returns></returns>
                    ''' <remarks>1 / (1 + Math.Exp(-Value))</remarks>
                    Private Shared Function Sigmoid(ByRef Value As Integer) As Double
                        'z = Bias + (Input*Weight)
                        'Output = 1/1+e**z
                        Return 1 / (1 + Math.Exp(-Value))
                    End Function

                    Private Shared Function SigmoidDerivitive(ByRef Value As Integer) As Double
                        Return Sigmoid(Value) * (1 - Sigmoid(Value))
                    End Function

                    Private Shared Function Signum(ByRef Value As Integer) As Double
                        'z = Bias + (Input*Weight)
                        'Output = 1/1+e**z
                        Return Math.Sign(Value)
                    End Function

                    Private Shared Function StochasticBinary(ByRef value As Double) As Double
                        'Uncreated
                        Return value
                    End Function

                End Structure

            End Class

            ''' <summary>
            ''' Activation function used by the nodes in the layer
            ''' </summary>
            ''' <remarks></remarks>
            Public ActivationFunction As TransferFunctionType

            ''' <summary>
            ''' Usually 1/0
            ''' </summary>
            Public Bias As Integer

            ''' <summary>
            ''' Calculates Layer Error From Output Vector
            ''' each scalar error output of the vector respective of its output
            ''' </summary>
            Public LayerError As Vector

            ''' <summary>
            ''' Type of layer (Input, Hidden, Output)
            ''' </summary>
            ''' <remarks></remarks>
            Public nLayerType As LayerType

            ''' <summary>
            ''' Collection of nodes
            ''' </summary>
            ''' <remarks></remarks>
            Public Nodes As List(Of Neuron)

            ''' <summary>
            ''' The number of nodes is stored to make iteration easier
            ''' </summary>
            ''' <remarks></remarks>
            Public ReadOnly Property NumberOfNodes As Integer
                Get
                    If Nodes IsNot Nothing Then
                        NumberOfNodes = Nodes.Count
                    Else
                        NumberOfNodes = 0
                    End If
                End Get
            End Property

            ''' <summary>
            ''' Executes Layer (Forwards Prop)
            ''' </summary>
            ''' <param name="Input">input vector</param>
            ''' <returns>Output Vector for the layer</returns>
            Public Function Execute(ByRef Input As Vector) As Vector
                SetInput(Input)
                ActivateLayer()
                Return GetOutput()
            End Function

            ''' <summary>
            ''' Returns Output as vector (as held in nodes at current state)
            ''' </summary>
            ''' <returns></returns>
            Public Function GetOutput() As Vector
                Dim NewVect As New Vector(New List(Of Double))
                For Each ITEM In Nodes
                    NewVect.values.Add(ITEM.output)
                Next
                Return NewVect
            End Function

            ''' <summary>
            ''' Activates each node in the layer -
            ''' Also the Layer is summed then activated and the single returned value is returned to Layer - output
            ''' </summary>
            ''' <remarks>layer to be summed to be passed to the inputs of the next layer</remarks>
            Private Sub ActivateLayer()
                Dim LayerTotal As Double = 0
                For Each node As Neuron In Nodes
                    'Sum Layer (NodeTotal = Input*Weight)
                    LayerTotal += node.NodeTotal()
                    'Activate Output (F(SumOfWeightedInputs)
                    node.output = TransferFunction.EvaluateTransferFunct(TransferFunctionType.Sigmoid, LayerTotal) + Bias
                Next
            End Sub

            ''' <summary>
            ''' Sets input for the Layer:
            ''' If input and nodes do not match will not be added
            ''' </summary>
            ''' <param name="Input"></param>
            Private Sub SetInput(ByRef Input As Vector)
                If Nodes.Count <> Input.values.Count Then

                    'Skip
                Else
                    For i = 0 To Nodes.Count
                        Nodes(i).SetInput(Input.values(i))
                    Next
                End If
            End Sub

            ''' <summary>
            ''' Each layer consists of neurons(nodes) the training cases also use an input layer and an
            ''' output layer
            ''' </summary>
            ''' <remarks></remarks>
            Public Sub New()
                Nodes = New List(Of Neuron)
                Bias = 1
            End Sub

            Public Shared Function CreateLayer(ByRef nLayertype As LayerType, ByRef NodesNo As Integer, ByRef Activation As TransferFunctionType) As Layer
                Dim layr As New Layer
                layr.nLayerType = nLayertype
                layr.ActivationFunction = Activation
                For i = 1 To NodesNo
                    Dim nde As New Neuron
                    nde.CreateRandWeight(0, 1)
                    layr.Nodes.Add(nde)
                Next
                Return layr
            End Function

            ''' <summary>
            ''' deserialize object from Json
            ''' </summary>
            ''' <param name="Str">json</param>
            ''' <returns></returns>
            Public Shared Function FromJson(ByRef Str As String) As Layer
                Try
                    Dim Converter As New JavaScriptSerializer
                    Dim diag As Layer = Converter.Deserialize(Of Layer)(Str)
                    Return diag
                Catch ex As Exception
                    Dim Buttons As MessageBoxButtons = MessageBoxButtons.OK
                    MessageBox.Show(ex.Message, "ERROR", Buttons)
                End Try
                Return Nothing
            End Function

            ''' <summary>
            ''' Serializes object to json
            ''' </summary>
            ''' <returns> </returns>
            Public Function ToJson() As String
                Dim Converter As New JavaScriptSerializer
                Return Converter.Serialize(Me)
            End Function

            ''' <summary>
            ''' Given the desired / expected output vector for the layer
            ''' the internal error is calculated as LAYERERRROR
            ''' Each Neuron Error for the layer is also produced
            ''' </summary>
            ''' <param name="DesiredOutput"> Each Output Error Vector value  Corresponds to a node error
            ''' Its error value is passed to calculate the individual neuron error</param>
            Public Sub Recalculate(ByRef DesiredOutput As Vector)

                'Calculate Node Errors for the layer
                'Each Output Error Vector Corresponds to a node
                'Its error value is passed to calculate the individual neuron error recalculating its new weight

                Dim cnt As Integer = 0

                For Each item In GetOutput.values

                    For Each nde In Nodes
                        nde.RecalculateWeight(GetOutput.values(cnt), DesiredOutput.values(cnt))
                    Next
                    cnt += 1
                Next
            End Sub

        End Class

        'Dim Delta As Double = learningRate * (NodeOutput - ExpectedOutput) * ExpectedOutput * DerivativeOfNodeOutput

        ''' <summary>
        ''' This is the Allowed Error threshold for the output,
        ''' The output could ba a single value or a vector
        ''' </summary>
        Public ErrorThreshold As Double

        Public iNetworkType As NetworkType = NetworkType.None

        ''' <summary>
        ''' Middle layer: This layer is the real thing behind the network. Without this layer,
        ''' network would not be capable of solving complex problems. There can be any number or
        ''' middle or hidden layers. But, for most of the tasks, one is sufficient. The number
        ''' of neurons in this layer is crucial. There is no formula for calculating the number,
        ''' just hit and trial works. This layer takes the input from input layer, does some
        ''' calculations and forwards to the next layer, in most cases it is the output layer.
        ''' </summary>
        ''' <remarks>in a deep belief network there can be many hidden layers</remarks>
        Public HiddenLayers As List(Of Layer)

        ''' <summary>
        ''' layer takes the inputs(the values you pass) and forwards it to hidden layer. You can
        ''' just imagine input layer as a group of neurons whose sole task is to pass the
        ''' numeric inputs to the next level. Input layer never processes data, it just hands
        ''' over it.
        ''' </summary>
        ''' <remarks>there is only one layer for the input</remarks>
        Public InputLayer As Layer

        ''' <summary>
        ''' Output layer: This layer consists of neurons which output the result to you. This
        ''' layer takes the value from the previous layer, does calculations and gives the final
        ''' result. Basically, this layer is just like hidden layer but instead of passing
        ''' values to the next layer, the values are treated as output.
        ''' </summary>
        ''' <remarks>there is only one layer for the output</remarks>
        Public OutputLayer As Layer

        Public Sub New()
            HiddenLayers = New List(Of Layer)
        End Sub

        ''' <summary>
        ''' The number of hidden nodes to become effective is actually unknown yet a simple
        ''' calculation can be used to determine an initial value which should be effective;
        ''' </summary>
        ''' <param name="NumbeOfInputNodes">the number of input node used in the network</param>
        ''' <param name="NumberOfOutputNodes">the number of out put nodes in the network</param>
        ''' <returns>a reasonable calculation for hidden nodes</returns>
        ''' <remarks>
        ''' Deep layer networks have multiple hidden layers with varied number of nodes
        ''' </remarks>
        Private Function CalculateNumberOfHiddenNodes(ByRef NumbeOfInputNodes As Integer, ByRef NumberOfOutputNodes As Integer) As Integer
            CalculateNumberOfHiddenNodes = NumbeOfInputNodes + NumberOfOutputNodes / 2
            If CalculateNumberOfHiddenNodes < NumberOfOutputNodes Then CalculateNumberOfHiddenNodes = NumberOfOutputNodes
        End Function

        ''' <summary>
        ''' Create Neural Network
        ''' </summary>
        ''' <param name="InputNodes">number of required nodes</param>
        ''' <param name="OutputNodes">Number of required nodes</param>
        ''' <param name="InputTransferFunction">required transfer function</param>
        ''' <param name="OutputFunction">output transfer function</param>
        ''' <param name="ErrThreshold">threshold error measurement (used for training network)</param>
        Public Sub New(ByRef InputNodes As Integer, OutputNodes As Integer,
                       ByRef InputTransferFunction As TransferFunctionType, ByRef OutputFunction As TransferFunctionType,
                       ByRef ErrThreshold As Double)
            ErrorThreshold = ErrThreshold
            Dim NoHidden As Integer = CalculateNumberOfHiddenNodes(InputNodes, OutputNodes)
            InputLayer = New Layer
            InputLayer = Layer.CreateLayer(LayerType.Input, InputNodes, InputTransferFunction)
            HiddenLayers = New List(Of Layer)
            For i = 1 To NoHidden
                HiddenLayers.Add(Layer.CreateLayer(LayerType.Hidden, InputNodes, TransferFunctionType.Sigmoid))
            Next
            OutputLayer = New Layer
            OutputLayer = Layer.CreateLayer(LayerType.Output, OutputNodes, OutputFunction)
        End Sub
        Public Enum LayerType
            Input
            Hidden
            Output
        End Enum
        ''' <summary>
        ''' Executes Networks (Single Iteration of Neural Network)
        ''' </summary>
        ''' <param name="Input">Input vector</param>
        ''' <returns>Output Vector</returns>
        Public MustOverride Function Execute(ByRef Input As Vector) As Vector

        ''' <summary>
        ''' Executes Networks (Single Iteration of Neural Network)
        ''' </summary>
        ''' <param name="Input">Input vector</param>
        ''' <returns>Output Vector</returns>
        Public MustOverride Function Train(ByRef Input As Vector) As Vector

        ''' <summary>
        ''' deserialize object from Json
        ''' </summary>
        ''' <param name="Str">json</param>
        ''' <returns></returns>
        Public Shared Function FromJson(ByRef Str As String) As NeuralNetworkFactory
            Try
                Dim Converter As New JavaScriptSerializer
                Dim diag As NeuralNetworkFactory = Converter.Deserialize(Of NeuralNetworkFactory)(Str)
                Return diag
            Catch ex As Exception
                Dim Buttons As MessageBoxButtons = MessageBoxButtons.OK
                MessageBox.Show(ex.Message, "ERROR", Buttons)
            End Try
            Return Nothing
        End Function

        ''' <summary>
        ''' Serializes object to json
        ''' </summary>
        ''' <returns> </returns>
        Public Function ToJson() As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(Me)
        End Function

    End Class

    ''' <summary>
    ''' The Perceptron Allows for a multi input vector to single output
    ''' </summary>
    Public Class Perceptron

        Public Property Weights As Double() ' The weights of the perceptron

        Private Function Sigmoid(x As Double) As Double ' The sigmoid activation function

            Return 1 / (1 + Math.Exp(-x))
        End Function

        ''' <summary>
        ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
        ''' binary data.
        ''' </summary>
        ''' <param name="Value"></param>
        ''' <returns></returns>
        ''' <remarks></remarks>
        Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

            ' Z = Bias+ (Input*Weight)
            'TransferFunction
            'If Z > 0 then Y = 1
            'If Z < 0 then y = 0

            Return If(Value < 0 = True, 0, 1)
        End Function



        Public Sub New(NumberOfInputs As Integer) ' Constructor that initializes the weights and bias of the perceptron
            CreateWeights(NumberOfInputs)

        End Sub

        Public Sub CreateWeights(NumberOfInputs As Integer) ' Constructor that initializes the weights and bias of the perceptron
            Weights = New Double(NumberOfInputs - 1) {}
            For i As Integer = 0 To NumberOfInputs - 1
                Weights(i) = Rnd(1.0)
            Next

        End Sub

        ' Function to calculate output
        Public Function Compute(inputs As Double()) As Integer
            CreateWeights(inputs.Count)
            Dim sum = 0.0

            ' Loop through inputs and calculate sum of weights times inputs
            For i = 0 To inputs.Length - 1
                sum += _Weights(i) * inputs(i)
            Next

            ' Return 1 if sum is greater than 0, otherwise return -1
            Return If(sum > 0, 1, -1)
        End Function

        Public Function ComputeSigmoid(inputs As Double()) As Double ' Compute the output of the perceptron given an input
            CreateWeights(inputs.Count)
            Dim sum As Double = 0
            'Collect the sum of the inputs * Weight
            For i As Integer = 0 To inputs.Length - 1
                sum += inputs(i) * Weights(i)
            Next

            'Activate
            'We Return the sigmoid of the sum to produce the output
            Return Sigmoid(sum)
        End Function

        Public Function ComputeBinaryThreshold(inputs As Double()) As Double ' Compute the output of the perceptron given an input
            CreateWeights(inputs.Count)
            Dim sum As Double = 0 ' used to hold the output

            'Collect the sum of the inputs * Weight
            For i As Integer = 0 To inputs.Length - 1
                sum += inputs(i) * Weights(i)
            Next

            'Activate
            'We Return the sigmoid of the sum to produce the output , Applying the Binary threshold funciton to it
            Return BinaryThreshold(Sigmoid(sum))
        End Function

        ' Function to train the perceptron
        Public Sub Train(inputs As Double(), desiredOutput As Integer, threshold As Double, MaxEpochs As Integer, LearningRate As Double)
            Dim guess = Compute(inputs)
            Dim nError As Integer = 0
            Dim CurrentEpoch = 0

            Do Until threshold < nError Or
                        CurrentEpoch = MaxEpochs
                CurrentEpoch += 1

                nError = desiredOutput - guess

                ' Loop through inputs and update weights based on error and learning rate
                For i = 0 To inputs.Length - 1
                    _Weights(i) += LearningRate * nError * inputs(i)
                Next

            Loop

        End Sub

    End Class

    Public Module NN_tests

        ''' <summary>
        ''' Here a single perceptron is used as a Layer
        ''' </summary>
        ''' <returns></returns>
        Public Property Layers As List(Of Perceptron) ' The layers of the network

        Public Function ComputePerceptronLayer(inputs As Double()) As Double ' Compute the output of the network given an input
            Dim output As Double() = inputs
            For Each layer In Layers
                Dim newOutput(layer.Weights.Length - 1) As Double
                For i As Integer = 0 To layer.Weights.Length - 1
                    newOutput(i) = layer.ComputeSigmoid(output)
                Next
                output = newOutput
            Next
            Return output(0)
        End Function

        Public Sub TrainBackProp(inputs As Double()(), outputs As Double(), Optional learningRate As Double = 0.1) ' Train the network given a set of inputs and outputs
            Dim errorThreshold As Double = 0.01 ' The error threshold at which to stop training
            Dim nError As Double = 1 ' Initialize the error to a high value
            While nError > errorThreshold ' Loop until the error is below the threshold
                nError = 0 ' Reset the error to zero
                For i As Integer = 0 To inputs.Length - 1 ' Loop through each input/output pair
                    Dim output As Double = ComputePerceptronLayer(inputs(i)) ' Compute the output of the network for this input
                    Dim delta As Double = learningRate * (outputs(i) - output) * output * (1 - output) ' Compute the delta for each weight and bias in the network
                    '   Dim Delta2 As Double = learningRate * (NodeOutput - ExpectedOutput) * ExpectedOutput * DerivativeOfOutput
                    For Each layer In Layers ' Loop through each layer in the network
                        For j As Integer = 0 To layer.Weights.Length - 1 ' Loop through each weight in this layer
                            layer.Weights(j) += delta * layer.ComputeSigmoid(inputs(i)) ' Update the weight based on this input and delta
                        Next
                        ' layer.Bias += delta ' Update the bias based on delta
                    Next
                    nError += Math.Abs(outputs(i) - output) ' Add the absolute difference between the actual output and predicted output to the total error
                Next
                nError /= inputs.Length ' Divide the total error by the number of input/output pairs to get the average error
            End While
        End Sub

    End Module


    Module helper
        Public Function MultiplyMatrix(matrixA As Double(,), matrixB As Double(,)) As Double(,)
            Dim rowsA As Integer = matrixA.GetLength(0)
            Dim columnsA As Integer = matrixA.GetLength(1)
            Dim rowsB As Integer = matrixB.GetLength(0)
            Dim columnsB As Integer = matrixB.GetLength(1)

            If columnsA <> rowsB Then
                Throw New ArgumentException("Invalid matrix dimensions for multiplication.")
            End If

            Dim resultMatrix As Double(,) = New Double(rowsA - 1, columnsB - 1) {}
            For i As Integer = 0 To rowsA - 1
                For j As Integer = 0 To columnsB - 1
                    Dim sum As Double = 0
                    For k As Integer = 0 To columnsA - 1
                        sum += matrixA(i, k) * matrixB(k, j)
                    Next
                    resultMatrix(i, j) = sum
                Next
            Next

            Return resultMatrix
        End Function
        Public Sub PrintMatrix(matrix As Double(,))
            Dim rows As Integer = matrix.GetLength(0)
            Dim columns As Integer = matrix.GetLength(1)

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To columns - 1
                    Console.Write(matrix(i, j) & " ")
                Next
                Console.WriteLine()
            Next
            Console.WriteLine()
        End Sub


    End Module
    Public Class Tril
        Public Sub Main()
            Dim matrix(,) As Integer = {{1, 2, 3, 9}, {4, 5, 6, 8}, {7, 8, 9, 9}}

            Dim result(,) As Integer = Tril(matrix)

            Console.WriteLine("Matrix:")
            PrintMatrix(matrix)

            Console.WriteLine("Tril Result:")
            PrintMatrix(result)
            Console.ReadLine()
        End Sub

        Public Shared Function Tril(ByVal matrix(,) As Integer) As Integer(,)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            Dim result(rows - 1, cols - 1) As Integer

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i, j) = matrix(i, j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Function Tril(ByVal matrix(,) As Double) As Double(,)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            Dim result(rows - 1, cols - 1) As Double

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i, j) = matrix(i, j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Function Tril(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim rows As Integer = matrix.Count
            Dim cols As Integer = matrix(0).Count

            Dim result As New List(Of List(Of Double))

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i)(j) = matrix(i)(j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Sub PrintMatrix(ByVal matrix(,) As Double)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    Console.Write(matrix(i, j) & " ")
                Next
                Console.WriteLine()
            Next
        End Sub
        Public Shared Sub PrintMatrix(ByVal matrix(,) As Integer)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    Console.Write(matrix(i, j) & " ")
                Next
                Console.WriteLine()
            Next
        End Sub
    End Class
    Public Class Softmax
        Public Shared Function Softmax(matrix2 As Integer(,)) As Double(,)
            Dim numRows As Integer = matrix2.GetLength(0)
            Dim numColumns As Integer = matrix2.GetLength(1)

            Dim softmaxValues(numRows - 1, numColumns - 1) As Double

            ' Compute softmax values for each row
            For i As Integer = 0 To numRows - 1
                Dim rowSum As Double = 0

                ' Compute exponential values and sum of row elements
                For j As Integer = 0 To numColumns - 1
                    softmaxValues(i, j) = Math.Sqrt(Math.Exp(matrix2(i, j)))
                    rowSum += softmaxValues(i, j)
                Next

                ' Normalize softmax values for the row
                For j As Integer = 0 To numColumns - 1
                    softmaxValues(i, j) /= rowSum
                Next
            Next

            ' Display the softmax values
            Console.WriteLine("Calculated:" & vbNewLine)
            For i As Integer = 0 To numRows - 1
                For j As Integer = 0 To numColumns - 1

                    Console.Write(softmaxValues(i, j).ToString("0.0000") & " ")
                Next
                Console.WriteLine(vbNewLine & "---------------------")
            Next
            Return softmaxValues
        End Function
        Public Shared Sub Main()
            Dim input() As Double = {1.0, 2.0, 3.0}

            Dim output() As Double = Softmax(input)

            Console.WriteLine("Input: {0}", String.Join(", ", input))
            Console.WriteLine("Softmax Output: {0}", String.Join(", ", output))
            Console.ReadLine()
        End Sub

        Public Shared Function Softmax(ByVal input() As Double) As Double()
            Dim maxVal As Double = input.Max()

            Dim exponentiated() As Double = input.Select(Function(x) Math.Exp(x - maxVal)).ToArray()

            Dim sum As Double = exponentiated.Sum()

            Dim softmaxOutput() As Double = exponentiated.Select(Function(x) x / sum).ToArray()

            Return softmaxOutput
        End Function
    End Class
    Public Class Vector
        Public ReadOnly values As List(Of Double)
        Public Function ApplyActivationFunction() As Vector
            ' Apply the desired activation function to each value in the vector
            Dim result As New List(Of Double)()

            For Each value In values
                result.Add(Sigmoid(value)) ' Applying Sigmoid activation function
            Next

            Return New Vector(result)
        End Function

        Private Function Sigmoid(x As Double) As Double
            Return 1 / (1 + Math.Exp(-x))
        End Function
        Public Sub New(values As List(Of Double))
            Me.values = values
        End Sub

        Public Function Add(other As Vector) As Vector
            If values.Count <> other.values.Count Then
                Throw New ArgumentException("Vector dimensions do not match.")
            End If

            Dim result As New List(Of Double)()

            For i = 0 To values.Count - 1
                result.Add(values(i) + other.values(i))
            Next

            Return New Vector(result)
        End Function

        Public Function Subtract(other As Vector) As Vector
            If values.Count <> other.values.Count Then
                Throw New ArgumentException("Vector dimensions do not match.")
            End If

            Dim result As New List(Of Double)()

            For i = 0 To values.Count - 1
                result.Add(values(i) - other.values(i))
            Next

            Return New Vector(result)
        End Function

        Public Function Multiply(scalar As Double) As Vector
            Dim result As New List(Of Double)()

            For Each value In values
                result.Add(value * scalar)
            Next

            Return New Vector(result)
        End Function

        Public Function DotProduct(other As Vector) As Double
            If values.Count <> other.values.Count Then
                Throw New ArgumentException("Vector dimensions do not match.")
            End If

            Dim result As Double = 0

            For i = 0 To values.Count - 1
                result += values(i) * other.values(i)
            Next

            Return result
        End Function

        Public Function Norm() As Double
            Dim sumOfSquares As Double = values.Sum(Function(value) value * value)
            Return Math.Sqrt(sumOfSquares)
        End Function

        Public Overrides Function ToString() As String
            Return $"[{String.Join(", ", values)}]"
        End Function

        ' New extended functionality

        Public Function ElementWiseMultiply(other As Vector) As Vector
            If values.Count <> other.values.Count Then
                Throw New ArgumentException("Vector dimensions do not match.")
            End If

            Dim result As New List(Of Double)()

            For i = 0 To values.Count - 1
                result.Add(values(i) * other.values(i))
            Next

            Return New Vector(result)
        End Function

        Public Function ElementWiseDivide(other As Vector) As Vector
            If values.Count <> other.values.Count Then
                Throw New ArgumentException("Vector dimensions do not match.")
            End If

            Dim result As New List(Of Double)()

            For i = 0 To values.Count - 1
                result.Add(values(i) / other.values(i))
            Next

            Return New Vector(result)
        End Function
    End Class
    Public Class Matrix
        Public ReadOnly data As Double(,)

        Public Sub New(matrixData As Double(,))
            data = matrixData
        End Sub
        Public Function MultiplyVector(vector As Vector) As Vector
            If data.GetLength(1) <> vector.values.Count Then
                Throw New ArgumentException("Matrix columns do not match vector dimensions.")
            End If

            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)
            Dim result(numRows - 1) As Double

            For i = 0 To numRows - 1
                Dim sum As Double = 0
                For j = 0 To numCols - 1
                    sum += data(i, j) * vector.values(j)
                Next
                result(i) = sum
            Next

            Return New Vector(result.ToList)
        End Function

        Public Function Transpose() As Matrix
            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)

            Dim result(numCols - 1, numRows - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(j, i) = data(i, j)
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Function Add(other As Matrix) As Matrix
            If data.GetLength(0) <> other.data.GetLength(0) OrElse data.GetLength(1) <> other.data.GetLength(1) Then
                Throw New ArgumentException("Matrix dimensions do not match.")
            End If

            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(i, j) = data(i, j) + other.data(i, j)
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Function Subtract(other As Matrix) As Matrix
            If data.GetLength(0) <> other.data.GetLength(0) OrElse data.GetLength(1) <> other.data.GetLength(1) Then
                Throw New ArgumentException("Matrix dimensions do not match.")
            End If

            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(i, j) = data(i, j) - other.data(i, j)
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Function Multiply(scalar As Double) As Matrix
            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(i, j) = data(i, j) * scalar
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Function Multiply(other As Matrix) As Matrix
            If data.GetLength(1) <> other.data.GetLength(0) Then
                Throw New ArgumentException("Matrix dimensions do not match for multiplication.")
            End If

            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = other.data.GetLength(1)
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    Dim sum As Double = 0
                    For k = 0 To data.GetLength(1) - 1
                        sum += data(i, k) * other.data(k, j)
                    Next
                    result(i, j) = sum
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Function ElementWiseMultiply(other As Matrix) As Matrix
            If data.GetLength(0) <> other.data.GetLength(0) OrElse data.GetLength(1) <> other.data.GetLength(1) Then
                Throw New ArgumentException("Matrix dimensions do not match.")
            End If

            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(i, j) = data(i, j) * other.data(i, j)
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Function ElementWiseDivide(other As Matrix) As Matrix
            If data.GetLength(0) <> other.data.GetLength(0) OrElse data.GetLength(1) <> other.data.GetLength(1) Then
                Throw New ArgumentException("Matrix dimensions do not match.")
            End If

            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(i, j) = data(i, j) / other.data(i, j)
                Next
            Next

            Return New Matrix(result)
        End Function

        Public Sub Display()
            Dim numRows As Integer = data.GetLength(0)
            Dim numCols As Integer = data.GetLength(1)

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    Console.Write(data(i, j).ToString("0.00") & " ")
                Next
                Console.WriteLine()
            Next
            Console.WriteLine()
        End Sub

        Public Shared Function GenerateRandomMatrix(numRows As Integer, numCols As Integer, minValue As Double, maxValue As Double) As Matrix
            Dim rand As New Random()
            Dim result(numRows - 1, numCols - 1) As Double

            For i = 0 To numRows - 1
                For j = 0 To numCols - 1
                    result(i, j) = rand.NextDouble() * (maxValue - minValue) + minValue
                Next
            Next

            Return New Matrix(result)
        End Function
    End Class
    Public Class Tensor
        Public values As Double()
        Public shape As Integer()



        Private Function GetSliceSize(sliceShape As Integer()) As Integer
            Dim sliceSize As Integer = 1
            For Each dimSize As Integer In sliceShape
                sliceSize *= dimSize
            Next

            Return sliceSize
        End Function
        Public Sub New(values As Double(), shape As Integer())
            Me.values = values
            Me.shape = shape
        End Sub
        Public Function GetValue(indices As Integer()) As Double
            Dim flattenedIndex As Integer = GetFlattenedIndex(indices)
            Return values(flattenedIndex)
        End Function
        Public Sub SetValue(indices As Integer(), value As Double)
            Dim flattenedIndex As Integer = GetFlattenedIndex(indices)
            values(flattenedIndex) = value
        End Sub
        Public Function GetShape() As Integer()
            Return shape
        End Function
        Public Function GetFlattenedIndex(indices As Integer()) As Integer
            Dim flattenedIndex As Integer = 0
            Dim stride As Integer = 1

            For i As Integer = shape.Length - 1 To 0 Step -1
                flattenedIndex += indices(i) * stride
                stride *= shape(i)
            Next i

            Return flattenedIndex
        End Function
        Public Function Reshape(newShape As Integer()) As Tensor
            If values.Length <> GetTotalSize() Then
                Throw New InvalidOperationException("Cannot reshape tensor. Number of elements does not match the new shape.")
            End If

            Dim reshapedValues As Double() = New Double(GetTotalSize() - 1) {}
            Array.Copy(values, reshapedValues, values.Length)

            Return New Tensor(reshapedValues, newShape)
        End Function
        Public Function GetTotalSize() As Integer
            Dim totalSize As Integer = 1
            For Each dimSize As Integer In shape
                totalSize *= dimSize
            Next

            Return totalSize
        End Function

        Public Function Add(tensor As Tensor) As Tensor
            ValidateShapes(tensor)

            Dim resultValues As Double() = New Double(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = values(i) + tensor.values(i)
            Next i

            Return New Tensor(resultValues, shape)
        End Function

        Public Function Subtract(tensor As Tensor) As Tensor
            ValidateShapes(tensor)

            Dim resultValues As Double() = New Double(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = values(i) - tensor.values(i)
            Next i

            Return New Tensor(resultValues, shape)
        End Function

        Public Function Multiply(tensor As Tensor) As Tensor
            ValidateShapes(tensor)

            Dim resultValues As Double() = New Double(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = values(i) * tensor.values(i)
            Next i

            Return New Tensor(resultValues, shape)
        End Function

        Public Function MultiplyScalar(scalar As Double) As Tensor
            Dim resultValues As Double() = New Double(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = values(i) * scalar
            Next i

            Return New Tensor(resultValues, shape)
        End Function

        Public Function Transpose() As Tensor
            If shape.Length <> 2 Then
                Throw New InvalidOperationException("Cannot transpose tensor. Only 2-dimensional tensors are supported.")
            End If

            Dim resultValues As Double() = New Double(values.Length - 1) {}
            Dim resultShape As Integer() = {shape(1), shape(0)}

            For i As Integer = 0 To shape(0) - 1
                For j As Integer = 0 To shape(1) - 1
                    resultValues(j * shape(0) + i) = values(i * shape(1) + j)
                Next j
            Next i

            Return New Tensor(resultValues, resultShape)
        End Function

        Public Function Tril() As Tensor
            If shape.Length <> 2 Then
                Throw New InvalidOperationException("Cannot apply tril operation. Only 2-dimensional tensors are supported.")
            End If

            Dim resultValues As Double() = New Double(values.Length - 1) {}
            Dim resultShape As Integer() = {shape(0), shape(1)}

            For i As Integer = 0 To shape(0) - 1
                For j As Integer = 0 To shape(1) - 1
                    If j <= i Then
                        resultValues(i * shape(1) + j) = values(i * shape(1) + j)
                    End If
                Next j
            Next i

            Return New Tensor(resultValues, resultShape)
        End Function

        Public Function Concat(tensor As Tensor, axis As Integer) As Tensor
            If axis < 0 Or axis >= shape.Length Then
                Throw New ArgumentException("Invalid axis value.")
            End If

            If shape.Length <> tensor.shape.Length Then
                Throw New ArgumentException("Tensor shapes do not match.")
            End If

            For i As Integer = 0 To shape.Length - 1
                If i = axis Then
                    If shape(i) + tensor.shape(i) <> shape(i) Then
                        Throw New ArgumentException("Tensor shapes do not match.")
                    End If
                ElseIf shape(i) <> tensor.shape(i) Then
                    Throw New ArgumentException("Tensor shapes do not match.")
                End If
            Next i

            Dim resultShape As Integer() = CType(shape.Clone(), Integer())
            resultShape(axis) += tensor.shape(axis)

            Dim resultSize As Integer = GetTotalSize()
            Dim resultValues As Double() = New Double(resultSize - 1) {}

            Dim stride As Integer = 1
            For i As Integer = shape.Length - 1 To 0 Step -1
                If i = axis Then
                    stride *= resultShape(i)
                End If
                If i <> axis Then
                    stride *= shape(i)
                End If
            Next i

            Dim tensorIndices As Integer() = New Integer(shape.Length - 1) {}

            For i As Integer = 0 To resultSize - 1
                Dim tensorIndex As Integer = i \ stride
                Dim tensorOffset As Integer = i Mod stride

                If tensorIndex < shape(axis) Then
                    resultValues(i) = values(i)
                Else
                    tensorIndices(axis) = tensorIndex - shape(axis)
                    For j As Integer = 0 To shape.Length - 1
                        If j <> axis Then
                            tensorIndices(j) = tensorOffset \ stride
                            tensorOffset = tensorOffset Mod stride
                            stride = stride \ shape(j)
                        End If
                    Next j

                    resultValues(i) = tensor.values(tensor.GetFlattenedIndex(tensorIndices))
                End If
            Next i

            Return New Tensor(resultValues, resultShape)
        End Function




        Public Overrides Function ToString() As String
            Dim result As String = ""
            Dim indices As Integer() = New Integer(shape.Length - 1) {}
            PrintTensorValues(result, indices, 0)

            Return result
        End Function

        Private Sub PrintTensorValues(ByRef result As String, indices As Integer(), dimension As Integer)
            Dim currentDimSize As Integer = shape(dimension)

            For i As Integer = 0 To currentDimSize - 1
                indices(dimension) = i

                If dimension = shape.Length - 1 Then
                    Dim flattenedIndex As Integer = GetFlattenedIndex(indices)
                    result += values(flattenedIndex).ToString() + " "
                Else
                    PrintTensorValues(result, indices, dimension + 1)
                End If

                If i < currentDimSize - 1 Then
                    result += ", "
                End If
            Next i

            If dimension = 0 Then
                result += Environment.NewLine
            End If
        End Sub

        Private Sub ValidateShapes(tensor As Tensor)
            If shape.Length <> tensor.shape.Length Then
                Throw New ArgumentException("Tensor shapes do not match.")
            End If

            For i As Integer = 0 To shape.Length - 1
                If shape(i) <> tensor.shape(i) Then
                    Throw New ArgumentException("Tensor shapes do not match.")
                End If
            Next i
        End Sub
    End Class
    Public Class Tensor(Of T)
        Private values As T()
        Private shape As Integer()
        Public Function Slice(startIndices As Integer(), endIndices As Integer()) As Tensor(Of T)
            Dim slicedShape As Integer() = New Integer(shape.Length - 1) {}
            Dim slicedValuesCount As Integer = 1

            For i As Integer = 0 To shape.Length - 1
                Dim startIdx As Integer = startIndices(i)
                Dim endIdx As Integer = endIndices(i)

                If startIdx < 0 OrElse startIdx >= shape(i) OrElse endIdx < 0 OrElse endIdx >= shape(i) Then
                    Throw New ArgumentException("Invalid slice indices.")
                End If

                slicedShape(i) = endIdx - startIdx + 1
                slicedValuesCount *= slicedShape(i)
            Next

            Dim slicedValues As T() = New T(slicedValuesCount - 1) {}
            Dim slicedIndices As Integer() = New Integer(shape.Length - 1) {}

            For i As Integer = 0 To slicedValuesCount - 1
                Dim valueIndex As Integer = GetFlattenedIndex(slicedIndices)
                slicedValues(i) = values(valueIndex)

                For j As Integer = shape.Length - 1 To 0 Step -1
                    slicedIndices(j) += 1
                    If slicedIndices(j) <= endIndices(j) Then
                        Exit For
                    End If
                    slicedIndices(j) = startIndices(j)
                Next
            Next

            Return New Tensor(Of T)(slicedValues, slicedShape)
        End Function

        Public Sub New(values As T(), shape As Integer())
            Me.values = values
            Me.shape = shape
        End Sub

        Public Function GetValue(indices As Integer()) As T
            Dim flattenedIndex As Integer = GetFlattenedIndex(indices)
            Return values(flattenedIndex)
        End Function

        Public Sub SetValue(indices As Integer(), value As T)
            Dim flattenedIndex As Integer = GetFlattenedIndex(indices)
            values(flattenedIndex) = value
        End Sub

        Public Function GetShape() As Integer()
            Return shape
        End Function

        Public Function GetFlattenedIndex(indices As Integer()) As Integer
            Dim flattenedIndex As Integer = 0
            Dim stride As Integer = 1

            For i As Integer = shape.Length - 1 To 0 Step -1
                flattenedIndex += indices(i) * stride
                stride *= shape(i)
            Next i

            Return flattenedIndex
        End Function

        Public Function Reshape(newShape As Integer()) As Tensor(Of T)
            If values.Length <> GetTotalSize() Then
                Throw New InvalidOperationException("Cannot reshape tensor. Number of elements does not match the new shape.")
            End If

            Dim reshapedValues As T() = New T(GetTotalSize() - 1) {}
            Array.Copy(values, reshapedValues, values.Length)

            Return New Tensor(Of T)(reshapedValues, newShape)
        End Function

        Public Function GetTotalSize() As Integer
            Dim totalSize As Integer = 1
            For Each dimSize As Integer In shape
                totalSize *= dimSize
            Next

            Return totalSize
        End Function

        Public Function Add(tensor As Tensor(Of T)) As Tensor(Of T)
            ValidateShapes(tensor)

            Dim resultValues As T() = New T(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = AddValues(values(i), tensor.values(i))
            Next i

            Return New Tensor(Of T)(resultValues, shape)
        End Function

        Public Function Subtract(tensor As Tensor(Of T)) As Tensor(Of T)
            ValidateShapes(tensor)

            Dim resultValues As T() = New T(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = SubtractValues(values(i), tensor.values(i))
            Next i

            Return New Tensor(Of T)(resultValues, shape)
        End Function

        Public Function Multiply(tensor As Tensor(Of T)) As Tensor(Of T)
            ValidateShapes(tensor)

            Dim resultValues As T() = New T(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = MultiplyValues(values(i), tensor.values(i))
            Next i

            Return New Tensor(Of T)(resultValues, shape)
        End Function

        Public Function Divide(tensor As Tensor(Of T)) As Tensor(Of T)
            ValidateShapes(tensor)

            Dim resultValues As T() = New T(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = DivideValues(values(i), tensor.values(i))
            Next i

            Return New Tensor(Of T)(resultValues, shape)
        End Function

        Public Function Power(exponent As Double) As Tensor(Of T)
            Dim resultValues As T() = New T(values.Length - 1) {}
            For i As Integer = 0 To values.Length - 1
                resultValues(i) = ExponentiateValue(values(i), exponent)
            Next i

            Return New Tensor(Of T)(resultValues, shape)
        End Function

        Private Sub ValidateShapes(tensor As Tensor(Of T))
            Dim tensorShape As Integer() = tensor.GetShape()
            If Not shape.SequenceEqual(tensorShape) Then
                Throw New ArgumentException("Tensor shapes do not match.")
            End If
        End Sub

        Private Function AddValues(value1 As T, value2 As T) As T
            Dim convert1 As IConvertible = TryCast(value1, IConvertible)
            Dim convert2 As IConvertible = TryCast(value2, IConvertible)

            If convert1 Is Nothing OrElse convert2 Is Nothing Then
                Throw New ArgumentException("Values cannot be converted to Double.")
            End If

            Dim doubleValue1 As Double = convert1.ToDouble(Nothing)
            Dim doubleValue2 As Double = convert2.ToDouble(Nothing)

            Dim resultValue As Double = doubleValue1 + doubleValue2
            Return DirectCast(Convert.ChangeType(resultValue, GetType(T)), T)
        End Function

        Private Function SubtractValues(value1 As T, value2 As T) As T
            Dim convert1 As IConvertible = TryCast(value1, IConvertible)
            Dim convert2 As IConvertible = TryCast(value2, IConvertible)

            If convert1 Is Nothing OrElse convert2 Is Nothing Then
                Throw New ArgumentException("Values cannot be converted to Double.")
            End If

            Dim doubleValue1 As Double = convert1.ToDouble(Nothing)
            Dim doubleValue2 As Double = convert2.ToDouble(Nothing)

            Dim resultValue As Double = doubleValue1 - doubleValue2
            Return DirectCast(Convert.ChangeType(resultValue, GetType(T)), T)
        End Function

        Private Function MultiplyValues(value1 As T, value2 As T) As T
            Dim convert1 As IConvertible = TryCast(value1, IConvertible)
            Dim convert2 As IConvertible = TryCast(value2, IConvertible)

            If convert1 Is Nothing OrElse convert2 Is Nothing Then
                Throw New ArgumentException("Values cannot be converted to Double.")
            End If

            Dim doubleValue1 As Double = convert1.ToDouble(Nothing)
            Dim doubleValue2 As Double = convert2.ToDouble(Nothing)

            Dim resultValue As Double = doubleValue1 * doubleValue2
            Return DirectCast(Convert.ChangeType(resultValue, GetType(T)), T)
        End Function

        Private Function DivideValues(value1 As T, value2 As T) As T
            Dim convert1 As IConvertible = TryCast(value1, IConvertible)
            Dim convert2 As IConvertible = TryCast(value2, IConvertible)

            If convert1 Is Nothing OrElse convert2 Is Nothing Then
                Throw New ArgumentException("Values cannot be converted to Double.")
            End If

            Dim doubleValue1 As Double = convert1.ToDouble(Nothing)
            Dim doubleValue2 As Double = convert2.ToDouble(Nothing)

            Dim resultValue As Double = doubleValue1 / doubleValue2
            Return DirectCast(Convert.ChangeType(resultValue, GetType(T)), T)
        End Function

        Private Function ExponentiateValue(value As T, exponent As Double) As T
            Dim convert1 As IConvertible = TryCast(value, IConvertible)
            If convert1 Is Nothing Then
                Throw New ArgumentException("Value cannot be converted to Double.")
            End If

            Dim doubleValue As Double = convert1.ToDouble(Nothing)
            Dim resultValue As Double = Math.Pow(doubleValue, exponent)
            Return DirectCast(Convert.ChangeType(resultValue, GetType(T)), T)
        End Function
    End Class
    Public Class FeedForwardNetwork
        Public Class Layer
            Public Neurons As List(Of Neuron)
            ''' <summary>
            ''' Serializes object to json
            ''' </summary>
            ''' <returns> </returns>
            Public Function ToJson() As String
                Dim Converter As New JavaScriptSerializer
                Return Converter.Serialize(Me)
            End Function

            ''' <summary>
            ''' Serializes Object to XML
            ''' </summary>
            ''' <param name="FileName"></param>
            Public Sub ToXML(ByRef FileName As String)
                Dim serialWriter As StreamWriter
                serialWriter = New StreamWriter(FileName)
                Dim xmlWriter As New XmlSerializer(Me.GetType())
                xmlWriter.Serialize(serialWriter, Me)
                serialWriter.Close()
            End Sub
            Public Sub New(size As Integer, inputsCount As Integer)
                Neurons = New List(Of Neuron)()

                For i As Integer = 0 To size - 1
                    Dim neuron As New Neuron()
                    neuron.Weight = New Double(inputsCount - 1) {}
                    Neurons.Add(neuron)
                Next
            End Sub
        End Class
        Public Class Neuron
            ''' <summary>
            ''' Serializes object to json
            ''' </summary>
            ''' <returns> </returns>
            Public Function ToJson() As String
                Dim Converter As New JavaScriptSerializer
                Return Converter.Serialize(Me)
            End Function

            ''' <summary>
            ''' Serializes Object to XML
            ''' </summary>
            ''' <param name="FileName"></param>
            Public Sub ToXML(ByRef FileName As String)
                Dim serialWriter As StreamWriter
                serialWriter = New StreamWriter(FileName)
                Dim xmlWriter As New XmlSerializer(Me.GetType())
                xmlWriter.Serialize(serialWriter, Me)
                serialWriter.Close()
            End Sub
            Public Weight As Double()
            Public Output As Double
            Public iError As Double
            Public Bias As Double
            Public Sub SetInput(input As Double)
                Output = input
            End Sub

            Public Sub ActivateNode(activation As TransferFunction.TransferFunctionType)
                Output = TransferFunction.Activate(Output, activation)
            End Sub
        End Class
        ''' <summary>
        ''' Serializes object to json
        ''' </summary>
        ''' <returns> </returns>
        Public Function ToJson() As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(Me)
        End Function

        ''' <summary>
        ''' Serializes Object to XML
        ''' </summary>
        ''' <param name="FileName"></param>
        Public Sub ToXML(ByRef FileName As String)
            Dim serialWriter As StreamWriter
            serialWriter = New StreamWriter(FileName)
            Dim xmlWriter As New XmlSerializer(Me.GetType())
            xmlWriter.Serialize(serialWriter, Me)
            serialWriter.Close()
        End Sub
        Public Layers As List(Of List(Of Neuron))
        Private ReadOnly network As FeedForwardNetwork
        Private ReadOnly learningRate As Double
        ''' <summary>
        ''' Used to intitiate network
        ''' </summary>
        ''' <param name="layerSizes"></param>
        Public Sub New(layerSizes As Integer())
            Layers = New List(Of List(Of Neuron))()

            ' Create the input layer
            Dim inputLayer As New List(Of Neuron)()
            For i As Integer = 0 To layerSizes(0) - 1
                inputLayer.Add(New Neuron())
            Next
            Layers.Add(inputLayer)

            ' Create the hidden layers and output layer
            For i As Integer = 1 To layerSizes.Length - 1
                Dim hiddenLayer As New List(Of Neuron)()
                For j As Integer = 0 To layerSizes(i) - 1
                    hiddenLayer.Add(New Neuron())
                Next
                Layers.Add(hiddenLayer)
            Next
        End Sub
        ''' <summary>
        ''' Used to train network
        ''' </summary>
        ''' <param name="network"></param>
        ''' <param name="learningRate"></param>
        Public Sub New(network As FeedForwardNetwork, learningRate As Double)
            Me.network = network
            Me.learningRate = learningRate
        End Sub
        ''' <summary>
        ''' Used to load a trained network
        ''' </summary>
        ''' <param name="Network"></param>
        Public Sub New(ByRef Network As FeedForwardNetwork)
            Network = Network
        End Sub
        Public Sub SetInput(layerIndex As Integer, inputValues As Double())
            Dim inputLayer As List(Of Neuron) = Layers(layerIndex)
            If inputValues.Length <> inputLayer.Count Then
                Throw New ArgumentException("Number of input values does not match the size of the input layer.")
            End If

            For i As Integer = 0 To inputValues.Length - 1
                inputLayer(i).SetInput(inputValues(i))
            Next
        End Sub
        Public Function GetOutput(layerIndex As Integer) As Double()
            Dim outputLayer As List(Of Neuron) = Layers(layerIndex)
            Dim outputValues(outputLayer.Count - 1) As Double

            For i As Integer = 0 To outputValues.Length - 1
                outputValues(i) = outputLayer(i).Output
            Next

            Return outputValues
        End Function
        Public Sub PropagateForward(activation As TransferFunction.TransferFunctionType)
            For i As Integer = 1 To Layers.Count - 1
                Dim currentLayer As List(Of Neuron) = Layers(i)
                Dim previousLayer As List(Of Neuron) = Layers(i - 1)

                For Each currentNeuron As Neuron In currentLayer
                    Dim totalInput As Double = 0

                    For j As Integer = 0 To previousLayer.Count - 1
                        totalInput += previousLayer(j).Output * currentNeuron.Weight(j)
                    Next

                    currentNeuron.SetInput(totalInput)
                    currentNeuron.ActivateNode(activation)
                Next
            Next
        End Sub
        Public Sub Train(inputs As Double()(), targets As Double()(), activation As TransferFunction.TransferFunctionType, epochs As Integer)
            If inputs.Length <> targets.Length Then
                Throw New ArgumentException("Number of input patterns does not match the number of target patterns.")
            End If

            For epoch As Integer = 1 To epochs
                Dim totalLoss As Double = 0

                For i As Integer = 0 To inputs.Length - 1
                    Dim inputPattern As Double() = inputs(i)
                    Dim targetPattern As Double() = targets(i)

                    ' Set the input values
                    network.SetInput(0, inputPattern)

                    ' Propagate the inputs forward through the network
                    network.PropagateForward(activation)

                    ' Get the predicted output
                    Dim outputPattern As Double() = network.GetOutput(network.Layers.Count - 1)

                    ' Calculate the loss
                    Dim loss As Double = 0
                    For j As Integer = 0 To outputPattern.Length - 1
                        loss += Math.Pow(targetPattern(j) - outputPattern(j), 2)
                    Next
                    totalLoss += loss

                    ' Backpropagate the error and update the weights
                    For j As Integer = network.Layers.Count - 1 To 1 Step -1
                        Dim currentLayer As List(Of Neuron) = network.Layers(j)
                        Dim previousLayer As List(Of Neuron) = network.Layers(j - 1)

                        For Each currentNeuron As Neuron In currentLayer
                            Dim ierror As Double

                            If j = network.Layers.Count - 1 Then
                                ' Output layer
                                ierror = (targetPattern(currentLayer.IndexOf(currentNeuron)) - currentNeuron.Output) *
                                        TransferFunction.Derivative(currentNeuron.Output, activation)
                            Else
                                ' Hidden layer
                                ierror = 0
                                For k As Integer = 0 To network.Layers(j + 1).Count - 1
                                    Dim nextNeuron As Neuron = network.Layers(j + 1)(k)
                                    ierror += nextNeuron.iError * nextNeuron.Weight(currentLayer.IndexOf(currentNeuron))
                                Next
                                ierror *= TransferFunction.Derivative(currentNeuron.Output, activation)
                            End If

                            currentNeuron.iError = ierror

                            ' Update the weights
                            For k As Integer = 0 To previousLayer.Count - 1
                                Dim previousNeuron As Neuron = previousLayer(k)
                                Dim weightDelta As Double = learningRate * ierror * previousNeuron.Output
                                currentNeuron.Weight(k) += weightDelta
                            Next
                        Next
                    Next
                Next

                ' Print the average loss for the epoch
                Dim averageLoss As Double = totalLoss / inputs.Length
                Console.WriteLine("Epoch {0}: Average Loss = {1}", epoch, averageLoss)
            Next
        End Sub
        Public Sub StochasticGradientDescent(inputs As Double()(), targets As Double()(), activation As TransferFunction.TransferFunctionType, epochs As Integer)
            If inputs.Length <> targets.Length Then
                Throw New ArgumentException("Number of input patterns does not match the number of target patterns.")
            End If

            Dim random As New Random()

            For epoch As Integer = 1 To epochs
                Dim totalLoss As Double = 0

                For i As Integer = 0 To inputs.Length - 1
                    Dim index As Integer = random.Next(0, inputs.Length) ' Randomly select a pattern

                    Dim inputPattern As Double() = inputs(index)
                    Dim targetPattern As Double() = targets(index)

                    ' Set the input values
                    network.SetInput(0, inputPattern)

                    ' Propagate the inputs forward through the network
                    network.PropagateForward(activation)

                    ' Get the predicted output
                    Dim outputPattern As Double() = network.GetOutput(network.Layers.Count - 1)

                    ' Calculate the loss
                    Dim loss As Double = 0
                    For j As Integer = 0 To outputPattern.Length - 1
                        loss += Math.Pow(targetPattern(j) - outputPattern(j), 2)
                    Next
                    totalLoss += loss

                    ' Backpropagate the error and update the weights
                    BackpropagateError(targetPattern, activation)
                    UpdateWeights()
                Next

                ' Print the average loss for the epoch
                Dim averageLoss As Double = totalLoss / inputs.Length
                Console.WriteLine("Epoch {0}: Average Loss = {1}", epoch, averageLoss)
            Next
        End Sub

        Public Sub MiniBatchStochasticGradientDescent(inputs As Double()(), targets As Double()(), activation As TransferFunction.TransferFunctionType, epochs As Integer, batchSize As Integer)
            If inputs.Length <> targets.Length Then
                Throw New ArgumentException("Number of input patterns does not match the number of target patterns.")
            End If

            Dim random As New Random()

            For epoch As Integer = 1 To epochs
                Dim totalLoss As Double = 0

                ' Shuffle the data
                Dim shuffledIndices(inputs.Length - 1) As Integer
                For i As Integer = 0 To inputs.Length - 1
                    shuffledIndices(i) = i
                Next
                ShuffleArray(shuffledIndices)

                ' Iterate over mini-batches
                For start As Integer = 0 To inputs.Length - 1 Step batchSize
                    Dim endIdx As Integer = Math.Min(start + batchSize - 1, inputs.Length - 1)

                    ' Initialize accumulated gradients
                    Dim accumulatedGradients As New Dictionary(Of Neuron, Double())

                    ' Process each pattern in the mini-batch
                    For i As Integer = start To endIdx
                        Dim index As Integer = shuffledIndices(i)

                        Dim inputPattern As Double() = inputs(index)
                        Dim targetPattern As Double() = targets(index)

                        ' Set the input values
                        network.SetInput(0, inputPattern)

                        ' Propagate the inputs forward through the network
                        network.PropagateForward(activation)

                        ' Get the predicted output
                        Dim outputPattern As Double() = network.GetOutput(network.Layers.Count - 1)

                        ' Calculate the loss
                        Dim loss As Double = 0
                        For j As Integer = 0 To outputPattern.Length - 1
                            loss += Math.Pow(targetPattern(j) - outputPattern(j), 2)
                        Next
                        totalLoss += loss

                        ' Accumulate gradients for each layer
                        BackpropagateError(targetPattern, activation, accumulatedGradients)
                    Next

                    ' Update weights using accumulated gradients
                    UpdateWeights(accumulatedGradients)
                Next

                ' Print the average loss for the epoch
                Dim averageLoss As Double = totalLoss / inputs.Length
                Console.WriteLine("Epoch {0}: Average Loss = {1}", epoch, averageLoss)
            Next
        End Sub
        Private Sub BackpropagateError(targetPattern As Double(), activation As TransferFunction.TransferFunctionType, Optional ByRef accumulatedGradients As Dictionary(Of Neuron, Double()) = Nothing)
            For j As Integer = network.Layers.Count - 1 To 1 Step -1
                Dim currentLayer As List(Of Neuron) = network.Layers(j)
                Dim previousLayer As List(Of Neuron) = network.Layers(j - 1)

                For Each currentNeuron As Neuron In currentLayer
                    Dim ierror As Double

                    If j = network.Layers.Count - 1 Then
                        ' Output layer
                        ierror = (targetPattern(currentLayer.IndexOf(currentNeuron)) - currentNeuron.Output) *
                                TransferFunction.Derivative(currentNeuron.Output, activation)
                    Else
                        ' Hidden layer
                        ierror = 0
                        For k As Integer = 0 To network.Layers(j + 1).Count - 1
                            Dim nextNeuron As Neuron = network.Layers(j + 1)(k)
                            ierror += nextNeuron.iError * nextNeuron.Weight(currentLayer.IndexOf(currentNeuron))
                        Next
                        ierror *= TransferFunction.Derivative(currentNeuron.Output, activation)
                    End If

                    currentNeuron.iError = ierror

                    ' Accumulate gradients for weight updates
                    If accumulatedGradients IsNot Nothing Then
                        For k As Integer = 0 To previousLayer.Count - 1
                            Dim previousNeuron As Neuron = previousLayer(k)
                            Dim gradient As Double = ierror * previousNeuron.Output

                            If accumulatedGradients.ContainsKey(currentNeuron) Then
                                accumulatedGradients(currentNeuron)(k) += gradient
                            Else
                                accumulatedGradients(currentNeuron) = New Double(previousLayer.Count - 1) {}
                                accumulatedGradients(currentNeuron)(k) = gradient
                            End If
                        Next
                    End If
                Next
            Next
        End Sub
        Private Sub UpdateWeights(Optional ByRef accumulatedGradients As Dictionary(Of Neuron, Double()) = Nothing)
            For j As Integer = network.Layers.Count - 1 To 1 Step -1
                Dim currentLayer As List(Of Neuron) = network.Layers(j)
                Dim previousLayer As List(Of Neuron) = network.Layers(j - 1)

                For Each currentNeuron As Neuron In currentLayer
                    If accumulatedGradients IsNot Nothing Then
                        Dim gradients As Double() = accumulatedGradients(currentNeuron)

                        For k As Integer = 0 To previousLayer.Count - 1
                            Dim weightDelta As Double = learningRate * gradients(k)
                            currentNeuron.Weight(k) += weightDelta
                        Next
                    Else
                        For k As Integer = 0 To previousLayer.Count - 1
                            Dim previousNeuron As Neuron = previousLayer(k)
                            Dim weightDelta As Double = learningRate * currentNeuron.iError * previousNeuron.Output
                            currentNeuron.Weight(k) += weightDelta
                        Next
                    End If

                    ' Update bias weight
                    Dim biasWeightDelta As Double = learningRate * currentNeuron.iError
                    currentNeuron.Bias += biasWeightDelta
                Next
            Next
        End Sub

        Private Sub ShuffleArray(array As Integer())
            Dim random As New Random()
            Dim n As Integer = array.Length

            While n > 1
                n -= 1
                Dim k As Integer = random.Next(n + 1)
                Dim value As Integer = array(k)
                array(k) = array(n)
                array(n) = value
            End While
        End Sub
    End Class
    ''' <summary>
    ''' Transfer Function used in the calculation of the following layer
    ''' </summary>
    Public Structure TransferFunction
        Public Enum TransferFunctionType
            Sigmoid
            HyperbolicTangent
            BinaryThreshold
            RectifiedLinear
            Logistic
            StochasticBinary
            Gaussian
            Signum
            None
        End Enum

        Public Shared Function Activate(input As Double, type As TransferFunctionType) As Double
            Select Case type
                Case TransferFunctionType.Sigmoid
                    Return 1 / (1 + Math.Exp(-input))
                Case TransferFunctionType.HyperbolicTangent
                    Return Math.Tanh(input)
                Case TransferFunctionType.BinaryThreshold
                    Return If(input >= 0, 1, 0)
                Case TransferFunctionType.RectifiedLinear
                    Return Math.Max(0, input)
                Case TransferFunctionType.Logistic
                    Return 1 / (1 + Math.Exp(-input))
                Case TransferFunctionType.StochasticBinary
                    Return If(input >= 0, 1, 0)
                Case TransferFunctionType.Gaussian
                    Return Math.Exp(-(input * input))
                Case TransferFunctionType.Signum
                    Return Math.Sign(input)
                Case Else
                    Throw New ArgumentException("Invalid transfer function type.")
            End Select
        End Function

        Public Shared Function Derivative(output As Double, type As TransferFunctionType) As Double
            Select Case type
                Case TransferFunctionType.Sigmoid
                    Return output * (1 - output)
                Case TransferFunctionType.HyperbolicTangent
                    Return 1 - (output * output)
                Case TransferFunctionType.BinaryThreshold
                    Return 1
                Case TransferFunctionType.RectifiedLinear
                    Return If(output > 0, 1, 0)
                Case TransferFunctionType.Logistic
                    Return output * (1 - output)
                Case TransferFunctionType.StochasticBinary
                    Return 1
                Case TransferFunctionType.Gaussian
                    Return -2 * output * Math.Exp(-(output * output))
                Case TransferFunctionType.Signum
                    Return 0
                Case Else
                    Throw New ArgumentException("Invalid transfer function type.")
            End Select
        End Function

        ''' <summary>
        ''' Returns a result from the transfer function indicated ; Non Derivative
        ''' </summary>
        ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
        ''' <param name="Input">Input value for node/Neuron</param>
        ''' <returns>result</returns>
        Public Shared Function EvaluateTransferFunct(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
            EvaluateTransferFunct = 0
            Select Case TransferFunct
                Case TransferFunctionType.None
                    Return Input
                Case TransferFunctionType.Sigmoid
                    Return Sigmoid(Input)
                Case TransferFunctionType.HyperbolicTangent
                    Return HyperbolicTangent(Input)
                Case TransferFunctionType.BinaryThreshold
                    Return BinaryThreshold(Input)
                Case TransferFunctionType.RectifiedLinear
                    Return RectifiedLinear(Input)
                Case TransferFunctionType.Logistic
                    Return Logistic(Input)
                Case TransferFunctionType.Gaussian
                    Return Gaussian(Input)
                Case TransferFunctionType.Signum
                    Return Signum(Input)
            End Select
        End Function

        ''' <summary>
        ''' Returns a result from the transfer function indicated ; Non Derivative
        ''' </summary>
        ''' <param name="TransferFunct">Indicator for Transfer function selection</param>
        ''' <param name="Input">Input value for node/Neuron</param>
        ''' <returns>result</returns>
        Public Shared Function EvaluateTransferFunctionDerivative(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
            EvaluateTransferFunctionDerivative = 0
            Select Case TransferFunct
                Case TransferFunctionType.None
                    Return Input
                Case TransferFunctionType.Sigmoid
                    Return SigmoidDerivitive(Input)
                Case TransferFunctionType.HyperbolicTangent
                    Return HyperbolicTangentDerivative(Input)
                Case TransferFunctionType.Logistic
                    Return LogisticDerivative(Input)
                Case TransferFunctionType.Gaussian
                    Return GaussianDerivative(Input)
            End Select
        End Function

        ''' <summary>
        ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
        ''' binary data.
        ''' </summary>
        ''' <param name="Value"></param>
        ''' <returns></returns>
        ''' <remarks></remarks>
        Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

            ' Z = Bias+ (Input*Weight)
            'TransferFunction
            'If Z > 0 then Y = 1
            'If Z < 0 then y = 0

            Return If(Value < 0 = True, 0, 1)
        End Function

        Private Shared Function Gaussian(ByRef x As Double) As Double
            Gaussian = Math.Exp((-x * -x) / 2)
        End Function

        Private Shared Function GaussianDerivative(ByRef x As Double) As Double
            GaussianDerivative = Gaussian(x) * (-x / (-x * -x))
        End Function

        Private Shared Function HyperbolicTangent(ByRef Value As Double) As Double
            ' TanH(x) = (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))

            Return Math.Tanh(Value)
        End Function

        Private Shared Function HyperbolicTangentDerivative(ByRef Value As Double) As Double
            HyperbolicTangentDerivative = 1 - (HyperbolicTangent(Value) * HyperbolicTangent(Value)) * Value
        End Function

        'Linear Neurons
        ''' <summary>
        ''' in a liner neuron the weight(s) represent unknown values to be determined the
        ''' outputs could represent the known values of a meal and the inputs the items in the
        ''' meal and the weights the prices of the individual items There are no hidden layers
        ''' </summary>
        ''' <remarks>
        ''' answers are determined by determining the weights of the linear neurons the delta
        ''' rule is used as the learning rule: Weight = Learning rate * Input * LocalError of neuron
        ''' </remarks>
        Private Shared Function Linear(ByRef value As Double) As Double
            ' Output = Bias + (Input*Weight)
            Return value
        End Function

        'Non Linear neurons
        Private Shared Function Logistic(ByRef Value As Double) As Double
            'z = bias + (sum of all inputs ) * (input*weight)
            'output = Sigmoid(z)
            'derivative input = z/weight
            'derivative Weight = z/input
            'Derivative output = output*(1-Output)
            'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

            Return 1 / 1 + Math.Exp(-Value)
        End Function

        Private Shared Function LogisticDerivative(ByRef Value As Double) As Double
            'z = bias + (sum of all inputs ) * (input*weight)
            'output = Sigmoid(z)
            'derivative input = z/weight
            'derivative Weight = z/input
            'Derivative output = output*(1-Output)
            'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

            Return Logistic(Value) * (1 - Logistic(Value))
        End Function

        Private Shared Function RectifiedLinear(ByRef Value As Double) As Double
            'z = B + (input*Weight)
            'If Z > 0 then output = z
            'If Z < 0 then output = 0
            If Value < 0 = True Then

                Return 0
            Else
                Return Value
            End If
        End Function

        ''' <summary>
        ''' the log-sigmoid function constrains results to the range (0,1), the function is
        ''' sometimes said to be a squashing function in neural network literature. It is the
        ''' non-linear characteristics of the log-sigmoid function (and other similar activation
        ''' functions) that allow neural networks to model complex data.
        ''' </summary>
        ''' <param name="Value"></param>
        ''' <returns></returns>
        ''' <remarks>1 / (1 + Math.Exp(-Value))</remarks>
        Private Shared Function Sigmoid(ByRef Value As Integer) As Double
            'z = Bias + (Input*Weight)
            'Output = 1/1+e**z
            Return 1 / (1 + Math.Exp(-Value))
        End Function

        Private Shared Function SigmoidDerivitive(ByRef Value As Integer) As Double
            Return Sigmoid(Value) * (1 - Sigmoid(Value))
        End Function

        Private Shared Function Signum(ByRef Value As Integer) As Double
            'z = Bias + (Input*Weight)
            'Output = 1/1+e**z
            Return Math.Sign(Value)
        End Function

        Private Shared Function StochasticBinary(ByRef value As Double) As Double
            'Uncreated
            Return value
        End Function

    End Structure


    Public Class RNN
        ''' <summary>
        ''' Serializes object to json
        ''' </summary>
        ''' <returns> </returns>
        Public Function ToJson() As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(Me)
        End Function

        ''' <summary>
        ''' Serializes Object to XML
        ''' </summary>
        ''' <param name="FileName"></param>
        Public Sub ToXML(ByRef FileName As String)
            Dim serialWriter As StreamWriter
            serialWriter = New StreamWriter(FileName)
            Dim xmlWriter As New XmlSerializer(Me.GetType())
            xmlWriter.Serialize(serialWriter, Me)
            serialWriter.Close()
        End Sub
        Private inputSize As Integer
        Private hiddenSize As Integer
        Private outputSize As Integer
        Private learningRate As Double
        Private maxIterations As Integer

        Private inputWeights As List(Of List(Of Double))
        Private hiddenWeights As List(Of List(Of Double))

        Public Sub New(ByVal inputSize As Integer, ByVal hiddenSize As Integer, ByVal outputSize As Integer, ByVal learningRate As Double, ByVal maxIterations As Integer)
            Me.inputSize = inputSize
            Me.hiddenSize = hiddenSize
            Me.outputSize = outputSize
            Me.learningRate = learningRate
            Me.maxIterations = maxIterations

            ' Initialize weights randomly
            Me.inputWeights = InitializeWeights(inputSize, hiddenSize)
            Me.hiddenWeights = InitializeWeights(hiddenSize, outputSize)
        End Sub

        Private Function InitializeWeights(ByVal inputSize As Integer, ByVal outputSize As Integer) As List(Of List(Of Double))
            Dim weights As List(Of List(Of Double)) = New List(Of List(Of Double))
            Dim random As Random = New Random()

            For i As Integer = 0 To inputSize - 1
                Dim row As List(Of Double) = New List(Of Double)
                For j As Integer = 0 To outputSize - 1
                    row.Add(random.NextDouble())
                Next
                weights.Add(row)
            Next

            Return weights
        End Function

        Public Sub Train(ByVal inputSequence As List(Of Double), ByVal targetOutput As List(Of Double))
            Dim iteration As Integer = 0
            Dim ierror As Double = Double.PositiveInfinity

            While iteration < maxIterations AndAlso ierror > 0.001
                ' Forward propagation
                Dim hiddenLayerOutput As List(Of Double) = CalculateLayerOutput(inputSequence, inputWeights)
                Dim outputLayerOutput As List(Of Double) = CalculateLayerOutput(hiddenLayerOutput, hiddenWeights)

                ' Backpropagation
                Dim outputError As List(Of Double) = SubtractVectors(targetOutput, outputLayerOutput)
                Dim outputDelta As List(Of Double) = MultiplyVectorByScalar(outputError, ActivationDerivative(outputLayerOutput))

                Dim hiddenError As List(Of Double) = DotProduct(outputDelta, TransposeMatrix(hiddenWeights))
                Dim hiddenDelta As List(Of Double) = MultiplyVectorByScalar(hiddenError, ActivationDerivative(hiddenLayerOutput))

                ' Update weights
                hiddenWeights = AddVectors(hiddenWeights, OuterProduct(hiddenLayerOutput, outputDelta))
                inputWeights = AddVectors(inputWeights, OuterProduct(inputSequence, hiddenDelta))

                ' Calculate total error
                ierror = CalculateTotalError(targetOutput, outputLayerOutput)

                iteration += 1
            End While
        End Sub
        Private Function CalculateLayerOutput(ByVal input As List(Of Double), ByVal weights As List(Of List(Of Double))) As List(Of Double)
            Dim weightedSum As List(Of Double) = DotProduct(input, weights)
            Return ActivationFunction(weightedSum)
        End Function
        Public Function Predict(ByVal inputSequence As List(Of Double)) As List(Of Double)
            Dim hiddenLayerOutput As List(Of Double) = CalculateLayerOutput(inputSequence, inputWeights)
            Dim outputLayerOutput As List(Of Double) = CalculateLayerOutput(hiddenLayerOutput, hiddenWeights)
            Return outputLayerOutput
        End Function
        Private Function ActivationFunction(ByVal vector As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()
            For Each val As Double In vector
                result.Add(Math.Tanh(val))
            Next
            Return result
        End Function
        Private Function ActivationDerivative(ByVal vector As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()
            For Each val As Double In vector
                result.Add(1 - Math.Tanh(val) ^ 2)
            Next
            Return result
        End Function

        Private Function DotProduct(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of List(Of Double))) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()

            For Each row As List(Of Double) In vector2
                Dim sum As Double = 0
                For i As Integer = 0 To vector1.Count - 1
                    sum += vector1(i) * row(i)
                Next
                result.Add(sum)
            Next

            Return result
        End Function

        Private Function TransposeMatrix(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For i As Integer = 0 To matrix(0).Count - 1
                Dim row As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To matrix.Count - 1
                    row.Add(matrix(j)(i))
                Next
                result.Add(row)
            Next

            Return result
        End Function

        Private Function OuterProduct(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of Double)) As List(Of List(Of Double))
            Dim result As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For Each val1 As Double In vector1
                Dim row As List(Of Double) = New List(Of Double)()
                For Each val2 As Double In vector2
                    row.Add(val1 * val2)
                Next
                result.Add(row)
            Next

            Return result
        End Function

        Private Function AddVectors(ByVal vector1 As List(Of List(Of Double)), ByVal vector2 As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For i As Integer = 0 To vector1.Count - 1
                Dim row As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To vector1(i).Count - 1
                    row.Add(vector1(i)(j) + vector2(i)(j))
                Next
                result.Add(row)
            Next

            Return result
        End Function

        Private Function SubtractVectors(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To vector1.Count - 1
                result.Add(vector1(i) - vector2(i))
            Next

            Return result
        End Function

        Private Function MultiplyVectorByScalar(ByVal vector As List(Of Double), ByVal scalar As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To vector.Count - 1
                result.Add(vector(i) * scalar(i))
            Next

            Return result
        End Function

        Private Function CalculateTotalError(ByVal targetOutput As List(Of Double), ByVal predictedOutput As List(Of Double)) As Double
            Dim totalError As Double = 0

            For i As Integer = 0 To targetOutput.Count - 1
                totalError += (targetOutput(i) - predictedOutput(i)) ^ 2
            Next

            Return totalError / 2
        End Function
    End Class
    Public Class LSTM
        ''' <summary>
        ''' Serializes object to json
        ''' </summary>
        ''' <returns> </returns>
        Public Function ToJson() As String
            Dim Converter As New JavaScriptSerializer
            Return Converter.Serialize(Me)
        End Function

        ''' <summary>
        ''' Serializes Object to XML
        ''' </summary>
        ''' <param name="FileName"></param>
        Public Sub ToXML(ByRef FileName As String)
            Dim serialWriter As StreamWriter
            serialWriter = New StreamWriter(FileName)
            Dim xmlWriter As New XmlSerializer(Me.GetType())
            xmlWriter.Serialize(serialWriter, Me)
            serialWriter.Close()
        End Sub
        Private inputSize As Integer
        Private hiddenSize As Integer
        Private outputSize As Integer
        Private learningRate As Double
        Private maxIterations As Integer

        Private inputWeights As List(Of List(Of Double))
        Private hiddenWeights As List(Of List(Of Double))

        Public Sub New(ByVal inputSize As Integer, ByVal hiddenSize As Integer, ByVal outputSize As Integer, ByVal learningRate As Double, ByVal maxIterations As Integer)
            Me.inputSize = inputSize
            Me.hiddenSize = hiddenSize
            Me.outputSize = outputSize
            Me.learningRate = learningRate
            Me.maxIterations = maxIterations

            ' Initialize weights randomly
            Me.inputWeights = InitializeWeights(inputSize, hiddenSize)
            Me.hiddenWeights = InitializeWeights(hiddenSize, outputSize)
        End Sub

        Private Function InitializeWeights(ByVal inputSize As Integer, ByVal outputSize As Integer) As List(Of List(Of Double))
            Dim weights As List(Of List(Of Double)) = New List(Of List(Of Double))()
            Dim random As Random = New Random()

            For i As Integer = 0 To inputSize - 1
                Dim row As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To outputSize - 1
                    row.Add(random.NextDouble())
                Next
                weights.Add(row)
            Next

            Return weights
        End Function

        Public Sub Train(ByVal inputSequence As List(Of Double), ByVal targetOutput As List(Of Double))
            Dim iteration As Integer = 0
            Dim ierror As Double = Double.PositiveInfinity

            While iteration < maxIterations AndAlso ierror > 0.001
                ' Forward propagation
                Dim hiddenLayerOutput As List(Of Double) = CalculateLayerOutput(inputSequence, inputWeights)
                Dim outputLayerOutput As List(Of Double) = CalculateLayerOutput(hiddenLayerOutput, hiddenWeights)

                ' Backpropagation
                Dim outputError As List(Of Double) = SubtractVectors(targetOutput, outputLayerOutput)
                Dim outputDelta As List(Of Double) = MultiplyVectorByScalar(outputError, ActivationDerivative(outputLayerOutput))

                Dim hiddenError As List(Of Double) = DotProduct(outputDelta, TransposeMatrix(hiddenWeights))
                Dim hiddenDelta As List(Of Double) = MultiplyVectorByScalar(hiddenError, ActivationDerivative(hiddenLayerOutput))

                ' Update weights
                hiddenWeights = AddVectors(hiddenWeights, OuterProduct(hiddenLayerOutput, outputDelta))
                inputWeights = AddVectors(inputWeights, OuterProduct(inputSequence, hiddenDelta))

                ' Calculate total error
                ierror = CalculateTotalError(targetOutput, outputLayerOutput)

                iteration += 1
            End While
        End Sub

        Public Function Predict(ByVal inputSequence As List(Of Double)) As List(Of Double)
            Dim hiddenLayerOutput As List(Of Double) = CalculateLayerOutput(inputSequence, inputWeights)
            Dim outputLayerOutput As List(Of Double) = CalculateLayerOutput(hiddenLayerOutput, hiddenWeights)
            Return outputLayerOutput
        End Function

        Private Function ActivationFunction(ByVal vector As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()
            For Each val As Double In vector
                result.Add(Math.Tanh(val))
            Next
            Return result
        End Function

        Private Function ActivationDerivative(ByVal vector As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()
            For Each val As Double In vector
                result.Add(1 - Math.Tanh(val) ^ 2)
            Next
            Return result
        End Function

        Private Function CalculateLayerOutput(ByVal input As List(Of Double), ByVal weights As List(Of List(Of Double))) As List(Of Double)
            Dim weightedSum As List(Of Double) = DotProduct(input, weights)
            Return ActivationFunction(weightedSum)
        End Function

        Private Function DotProduct(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of List(Of Double))) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()

            For Each row As List(Of Double) In vector2
                Dim sum As Double = 0
                For i As Integer = 0 To vector1.Count - 1
                    sum += vector1(i) * row(i)
                Next
                result.Add(sum)
            Next

            Return result
        End Function

        Private Function TransposeMatrix(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For i As Integer = 0 To matrix(0).Count - 1
                Dim row As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To matrix.Count - 1
                    row.Add(matrix(j)(i))
                Next
                result.Add(row)
            Next

            Return result
        End Function

        Private Function OuterProduct(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of Double)) As List(Of List(Of Double))
            Dim result As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For Each val1 As Double In vector1
                Dim row As List(Of Double) = New List(Of Double)()
                For Each val2 As Double In vector2
                    row.Add(val1 * val2)
                Next
                result.Add(row)
            Next

            Return result
        End Function

        Private Function AddVectors(ByVal vector1 As List(Of List(Of Double)), ByVal vector2 As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For i As Integer = 0 To vector1.Count - 1
                Dim row As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To vector1(i).Count - 1
                    row.Add(vector1(i)(j) + vector2(i)(j))
                Next
                result.Add(row)
            Next

            Return result
        End Function

        Private Function SubtractVectors(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To vector1.Count - 1
                result.Add(vector1(i) - vector2(i))
            Next

            Return result
        End Function

        Private Function MultiplyVectorByScalar(ByVal vector As List(Of Double), ByVal scalar As List(Of Double)) As List(Of Double)
            Dim result As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To vector.Count - 1
                result.Add(vector(i) * scalar(i))
            Next

            Return result
        End Function

        Private Function CalculateTotalError(ByVal targetOutput As List(Of Double), ByVal predictedOutput As List(Of Double)) As Double
            Dim totalError As Double = 0

            For i As Integer = 0 To targetOutput.Count - 1
                totalError += (targetOutput(i) - predictedOutput(i)) ^ 2
            Next

            Return totalError / 2
        End Function
    End Class
End Namespace




