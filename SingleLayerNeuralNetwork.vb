Namespace Basic_NLP
    Public Class SingleLayerNeuralNetwork

        Public Inputs As List(Of Double)
        Public InputWeights As List(Of List(Of Double))


        Public Outputs As List(Of Double)
        Private HiddenLayer As List(Of Double)
        Private hiddenNodes As Integer
        Private HiddenWeights As List(Of List(Of Double))
        Private inputNodes As Integer
        Private learningRate As Double
        Private maxIterations As Integer
        Private outputNodes As Integer
        Public Sub New(ByRef PreTrained As SingleLayerNeuralNetwork)
            Me.Inputs = New List(Of Double)
            Me.HiddenLayer = New List(Of Double)
            Me.Outputs = New List(Of Double)
            Me.inputNodes = PreTrained.inputNodes
            Me.outputNodes = PreTrained.outputNodes
            Me.hiddenNodes = PreTrained.hiddenNodes
            Me.InputWeights = PreTrained.InputWeights
            Me.HiddenWeights = PreTrained.HiddenWeights
            Me.learningRate = PreTrained.learningRate
            Me.maxIterations = PreTrained.maxIterations
        End Sub

        Public Sub New(ByRef NumberOfInputNodes As Integer, ByRef NumberOfOutputNodes As Integer)

            Dim NumberOfHiddenNodes As Integer = CalculateNumberOfHiddenNodes(NumberOfInputNodes, NumberOfOutputNodes)
            ' Initialize the input weights
            Me.InputWeights = New List(Of List(Of Double))()
            For i As Integer = 0 To NumberOfHiddenNodes - 1
                Dim weights As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To NumberOfInputNodes - 1
                    weights.Add(Rnd())
                Next
                Me.InputWeights.Add(weights)
            Next

            ' Initialize the hidden weights
            Me.HiddenWeights = New List(Of List(Of Double))()
            For i As Integer = 0 To NumberOfOutputNodes - 1
                Dim weights As List(Of Double) = New List(Of Double)()
                For j As Integer = 0 To NumberOfHiddenNodes - 1
                    weights.Add(Rnd())
                Next
                Me.HiddenWeights.Add(weights)
            Next
        End Sub

        Public Enum ConcatenationType
            Horizontal
            Vertical
        End Enum
        Enum TransferFunctionType
            Sigmoid
            Sigmoid_Derivative
            SoftMax
            Identity
            Relu
            BinaryThreshold
            HyperbolicTangent
            RectifiedLinear
            Logistic
            StochasticBinary
            Gaussian
            Signum
            None
        End Enum
        Public Shared Function AddResidualConnections(ByRef nInputs As List(Of List(Of Double)), ByRef AddQuery As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As New List(Of List(Of Double))

            For i As Integer = 0 To nInputs.Count - 1
                Dim outputRow As New List(Of Double)

                For j As Integer = 0 To nInputs(i).Count - 1
                    outputRow.Add(nInputs(i)(j) + AddQuery(i)(j))
                Next

                result.Add(outputRow)
            Next

            Return result
        End Function

        Public Shared Function ApplyLinearTransformation(inputs As List(Of Double), weights As List(Of List(Of Double)), bias As Integer) As List(Of Double)
            Dim outputs = New List(Of Double)

            For Each _Weight In weights
                Dim sum = 0.0
                Dim Current As Integer = 0

                sum += inputs(Current) * _Weight(Current)
                Current += 1
                sum = sum + bias
                outputs.Add(sum)
            Next

            Return outputs
        End Function

        ''' <summary>
        ''' Input to this layer will be repeated for each head ,
        '''Query , Key and value,
        '''On entry the Input should be
        '''the key should be transposed first then passed through the linear layer
        '''the weights should be shared between inputs.
        '''The output for each q/k/v matrix ... should be as follows
        '''Value Remains untouched ,
        '''The A Output is created by the dot-product
        '''                         (of the output for the key and query from the linear layer)
        '''This output is scaled by the dimension of the key_vector
        '''And finally passed through a soft-max . This output is to be ,
        '''DotProduct with the Value matrix outputted from the linear layer
        '''To produce the final output for the Attention Head.
        ''' </summary>
        ''' <param name="batchinput"></param>
        ''' <returns></returns>
        Public Shared Function Attention(ByRef batchinput As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim batchSize As Integer = batchinput.Count
            Dim inputSize As Integer = batchinput(0).Count
            '1.Create Q , K, V
            '-> Q)Pass Q through linear layer
            '-> K)Transpose K and Pass through liner layer
            '-> V)Copy Q to V
            Dim Linear As New LinearLayer(batchinput, inputSize, batchSize)
            Dim query As List(Of List(Of Double)) = Linear.Forward(batchinput)
            Dim key As List(Of List(Of Double)) = MultMath.TransposeMatrix(batchinput)
            key = Linear.Forward(key)
            Dim value As List(Of List(Of Double)) = query
            '  Create dotProduct of Key*Query
            ' -> Scale by K(Dimension)
            ' -> SoftMax = AttentionOutput
            Dim attentionOutput As List(Of List(Of Double)) = MultMath.Softmax(MultMath.ScaleMatrix(MultMath.DotProduct(query, key), key.Count))
            'Create DotProduct of (V)Value & AttentionOutput
            'Return Attention head

            Return MultMath.DotProduct(attentionOutput, value)
        End Function

        Public Shared Function ConcatenateMatrices(matrix1 As List(Of List(Of Double)), matrix2 As List(Of List(Of Double)), concatenateVertical As ConcatenationType) As List(Of List(Of Double))
            Dim concatenatedMatrix As New List(Of List(Of Double))

            If concatenateVertical = ConcatenationType.Vertical Then
                ' Vertical concatenation
                concatenatedMatrix.AddRange(matrix1)
                concatenatedMatrix.AddRange(matrix2)
            Else
                ' Horizontal concatenation

                ' Ensure the matrices have the same number of rows
                If matrix1.Count <> matrix2.Count Then
                    Throw New ArgumentException("Matrices must have the same number of rows.")
                End If

                ' Concatenate the rows of matrix1 and matrix2 side by side
                For rowIndex As Integer = 0 To matrix1.Count - 1
                    Dim concatenatedRow As New List(Of Double)
                    concatenatedRow.AddRange(matrix1(rowIndex))
                    concatenatedRow.AddRange(matrix2(rowIndex))
                    concatenatedMatrix.Add(concatenatedRow)
                Next
            End If

            Return concatenatedMatrix
        End Function

        Public Shared Function InitializeWeightMatrix(ByVal inputSize As Integer, ByVal outputSize As Integer) As List(Of List(Of Double))
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

        Public Shared Sub Main()
            ' Define the training data
            Dim trainingInputs As List(Of List(Of Double)) = New List(Of List(Of Double))()
            trainingInputs.Add(New List(Of Double)() From {0, 0})
            trainingInputs.Add(New List(Of Double)() From {0, 1})
            trainingInputs.Add(New List(Of Double)() From {1, 0})
            trainingInputs.Add(New List(Of Double)() From {1, 1})

            Dim trainingTargets As List(Of List(Of Double)) = New List(Of List(Of Double))()
            trainingTargets.Add(New List(Of Double)() From {0})
            trainingTargets.Add(New List(Of Double)() From {1})
            trainingTargets.Add(New List(Of Double)() From {1})
            trainingTargets.Add(New List(Of Double)() From {0})

            ' Create a single-layer neural network
            Dim neuralNetwork As SingleLayerNeuralNetwork = New SingleLayerNeuralNetwork(2, 1)
            Dim trainer As New Trainer(neuralNetwork)
            ' Set the learning rate and number of epochs
            Dim learningRate As Double = 0.1
            Dim numEpochs As Integer = 1000

            ' Train the neural network
            For epoch As Integer = 1 To numEpochs
                Dim totalLoss As Double = 0.0

                For i As Integer = 0 To trainingInputs.Count - 1
                    Dim inputs As List(Of Double) = trainingInputs(i)
                    Dim targets As List(Of Double) = trainingTargets(i)

                    totalLoss += trainer.TrainSoftMax(inputs, targets, learningRate)
                Next

                ' Print the average loss for the epoch
                Dim averageLoss As Double = totalLoss / trainingInputs.Count
                Console.WriteLine("Epoch: {0}, Loss: {1}", epoch, averageLoss)
            Next

            ' Test the trained neural network
            Dim testInputs As List(Of List(Of Double)) = trainingInputs
            Dim testTargets As List(Of List(Of Double)) = trainingTargets

            Console.WriteLine("Testing the neural network:")

            For i As Integer = 0 To testInputs.Count - 1
                Dim inputs As List(Of Double) = testInputs(i)
                Dim targets As List(Of Double) = testTargets(i)
                neuralNetwork = trainer.ExportModel
                Dim predictions As List(Of Double) = neuralNetwork.ForwardPreNormalized(inputs, SingleLayerNeuralNetwork.TransferFunctionType.SoftMax)
                Console.WriteLine("Input: [{0}], Target: [{1}], Prediction: [{2}]", String.Join(", ", inputs), String.Join(", ", targets), String.Join(", ", predictions))
            Next

            Console.ReadLine()
        End Sub

        Public Shared Function Mask(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
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


        Public Shared Function MultiHeadedAttention(ByRef Input As List(Of List(Of Double)), Optional HeadCount As Integer = 8) As List(Of List(Of Double))
            Dim headSize As Integer = Input(0).Count \ HeadCount
            Dim heads As New List(Of List(Of List(Of Double)))
            For i = 0 To HeadCount - 1
                heads.Add(Attention(Input))
            Next
            Dim Output As New List(Of List(Of Double))
            For Each item In heads
                Output = ConcatenateMatrices(Output, item, ConcatenationType.Vertical)
            Next
            ' Apply linear transformation to obtain final output
            Dim finalOutput As List(Of List(Of Double)) = New LinearLayer(Input, HeadCount * headSize, Input(0).Count).Forward(Output)

            Return finalOutput

        End Function

        Public Shared Function MultiHeadedAttention(ByRef Q As List(Of List(Of Double)), ByRef K As List(Of List(Of Double)), ByRef V As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim headSize As Integer = V(0).Count \ 3
            Dim heads As New List(Of List(Of List(Of Double)))
            heads.Add(Attention(Q))
            heads.Add(Attention(K))
            heads.Add(Attention(V))

            Dim Output As New List(Of List(Of Double))
            For Each item In heads
                Output = ConcatenateMatrices(Output, item, ConcatenationType.Vertical)
            Next
            ' Apply linear transformation to obtain final output
            Dim finalOutput As List(Of List(Of Double)) = New LinearLayer(Q, 3 * headSize, V(0).Count).Forward(Output)

            Return finalOutput

        End Function

        Public Function ExportModel() As SingleLayerNeuralNetwork
            Return Me
        End Function

        Public Function Forward(ByRef Inputs As List(Of Double), Activation As TransferFunctionType) As List(Of Double)
            Me.Inputs = Inputs
            ' Calculate the weighted sum of inputs in the hidden layer
            Me.HiddenLayer = CalculateWeightedSum(Inputs, InputWeights)

            ' Apply activation function to the hidden layer
            Me.HiddenLayer = MultMath.Activate(HiddenLayer, Activation)

            ' Calculate the weighted sum of the hidden layer in the output layer
            Me.Outputs = CalculateWeightedSum(HiddenLayer, HiddenWeights)

            ' Apply activation function to the output layer
            Me.Outputs = MultMath.Activate(Outputs, Activation)

            Return Outputs
        End Function

        Public Function ForwardHidden(inputs As List(Of Double), transferFunction As TransferFunctionType) As List(Of Double)
            Dim hiddenLayer As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To Me.hiddenNodes - 1
                Dim sum As Double = 0.0
                For j As Integer = 0 To Me.inputNodes - 1
                    sum += inputs(j) * Me.InputWeights(i)(j)
                Next

                hiddenLayer.Add(MultMath.Activate(sum, transferFunction))
            Next

            Return hiddenLayer
        End Function

        Public Function ForwardOutput(hiddenLayer As List(Of Double), transferFunction As TransferFunctionType) As List(Of Double)
            Dim outputLayer As List(Of Double) = New List(Of Double)()

            For i As Integer = 0 To Me.outputNodes - 1
                Dim sum As Double = 0.0
                For j As Integer = 0 To Me.hiddenNodes - 1
                    sum += hiddenLayer(j) * Me.HiddenWeights(i)(j)
                Next

                outputLayer.Add(MultMath.Activate(sum, transferFunction))
            Next

            Return outputLayer
        End Function

        Public Function ForwardPostNormalized(ByRef Inputs As List(Of Double), Activation As TransferFunctionType) As List(Of Double)
            Me.Inputs = Inputs

            ' Calculate the weighted sum of inputs in the hidden layer
            Me.HiddenLayer = CalculateWeightedSum(Inputs, InputWeights)

            ' Apply activation function to the hidden layer
            Me.HiddenLayer = MultMath.Activate(HiddenLayer, Activation)

            ' Calculate the weighted sum of the hidden layer in the output layer
            Me.Outputs = CalculateWeightedSum(HiddenLayer, HiddenWeights)

            ' Apply activation function to the output layer
            Me.Outputs = MultMath.Activate(Outputs, Activation)

            ' Normalize the output
            NormalizeOutput()

            Return Outputs
        End Function

        Public Function ForwardPreNormalized(ByRef inputs As List(Of Double), transferFunction As TransferFunctionType) As List(Of Double)
            ' Normalize inputs to the range [0, 1]
            Dim normalizedInputs As List(Of Double) = New List(Of Double)

            For Each eInput In inputs
                normalizedInputs.Add(eInput / 1.0)
            Next

            ' Perform forward pass
            Dim hiddenLayer As List(Of Double) = ForwardHidden(normalizedInputs, transferFunction)
            Dim outputLayer As List(Of Double) = ForwardOutput(hiddenLayer, transferFunction)

            Return outputLayer
        End Function

        Public Function Predict(ByVal inputSequence As List(Of Double)) As List(Of Double)
            Dim hiddenLayerOutput As List(Of Double) = CalculateLayerOutput(inputSequence, InputWeights)
            Dim outputLayerOutput As List(Of Double) = CalculateLayerOutput(hiddenLayerOutput, HiddenWeights)
            Return outputLayerOutput
        End Function



        Private Function CalculateHiddenErrors(outputErrors As List(Of Double)) As List(Of Double)
            Dim hiddenErrors As List(Of Double) = New List(Of Double)

            For i As Integer = 0 To HiddenLayer.Count - 1
                Dim ierror As Double = 0.0

                For j As Integer = 0 To outputErrors.Count - 1
                    ierror += outputErrors(j) * HiddenWeights(i)(j)
                Next

                hiddenErrors.Add(ierror)
            Next

            Return hiddenErrors
        End Function

        Private Function CalculateLayerOutput(ByVal input As List(Of Double), ByVal weights As List(Of List(Of Double))) As List(Of Double)
            Dim weightedSum As List(Of Double) = MultMath.DotProduct(input, weights)
            Return MultMath.ActivationFunction(weightedSum)
        End Function

        Private Function CalculateNumberOfHiddenNodes(ByRef NumberOfInputNodes As Integer, ByRef NumberOfOutputNodes As Integer) As Integer
            Dim calculatedNumberOfHiddenNodes As Integer = NumberOfInputNodes + NumberOfOutputNodes / 2
            If calculatedNumberOfHiddenNodes < NumberOfOutputNodes Then
                calculatedNumberOfHiddenNodes = NumberOfOutputNodes
            End If
            Return calculatedNumberOfHiddenNodes
        End Function
        Private Function CalculateTotalError(ByVal targetOutput As List(Of Double), ByVal predictedOutput As List(Of Double)) As Double
            Dim totalError As Double = 0

            For i As Integer = 0 To targetOutput.Count - 1
                totalError += (targetOutput(i) - predictedOutput(i)) ^ 2
            Next

            Return totalError / 2
        End Function
        Private Function CalculateTotalError(outputErrors As List(Of Double)) As Double
            Dim totalError As Double = 0.0

            For Each ierror In outputErrors
                totalError += Math.Pow(ierror, 2)
            Next

            Return totalError
        End Function

        Private Function CalculateWeightedSum(inputs As List(Of Double), weights As List(Of List(Of Double))) As List(Of Double)
            Dim weightedSum As List(Of Double) = New List(Of Double)

            For i As Integer = 0 To weights(0).Count - 1
                Dim sum As Double = 0.0
                For j As Integer = 0 To inputs.Count - 1
                    sum += inputs(j) * weights(j)(i)
                Next
                weightedSum.Add(sum)
            Next

            Return weightedSum
        End Function
        Private Sub NormalizeOutput()
            Dim maxOutput As Double = Outputs.Max()
            Dim minOutput As Double = Outputs.Min()

            For i As Integer = 0 To Outputs.Count - 1
                Outputs(i) = (Outputs(i) - minOutput) / (maxOutput - minOutput)
            Next
        End Sub
        Private Sub UpdateWeights(inputs As List(Of Double), hiddenLayer As List(Of Double), learningRate As Double, hiddenErrors As List(Of Double), outputErrors As List(Of Double))
            ' Update weights connecting the input layer to the hidden layer
            For i As Integer = 0 To InputWeights.Count - 1
                For j As Integer = 0 To inputs.Count - 1
                    InputWeights(j)(i) += learningRate * hiddenErrors(i) * inputs(j)
                Next
            Next

            ' Update weights connecting the hidden layer to the output layer
            For i As Integer = 0 To HiddenWeights.Count - 1
                For j As Integer = 0 To hiddenLayer.Count - 1
                    HiddenWeights(i)(j) += learningRate * outputErrors(i) * hiddenLayer(j)
                Next
            Next
        End Sub

        Public Class LinearLayer
            Public Bias As Integer = 1
            Public Inputs As List(Of List(Of Double))
            Private ibias As List(Of Double)
            Private iWeights As List(Of List(Of Double))
            Dim rand As New Random()
            Public Sub New(ByRef nInputs As List(Of List(Of Double)), inputSize As Integer, outputSize As Integer)
                ''Set Inputs
                Inputs = nInputs
                ''Set Random Weights
                iWeights = CreateRandomMatrix(Inputs.Count, Inputs(0).Count)
                Randomize()
                Weights = InitializeWeightMatrix(inputSize, outputSize)
                ibias = New List(Of Double)
                For i As Integer = 0 To outputSize - 1

                    ibias.Add(rand.Next(-1, 1.0))
                Next
            End Sub

            Public Enum ConcatenationType
                Horizontal
                Vertical
            End Enum

            'shared to enable hyper parameters to be set or remembered
            Public Property Weights As List(Of List(Of Double))
                Get
                    Return iWeights
                End Get
                Set(value As List(Of List(Of Double)))
                    iWeights = value
                End Set
            End Property

            Public Shared Function CreateRandomMatrix(rows As Integer, columns As Integer) As List(Of List(Of Double))
                Dim random As New Random()
                Dim matrix As New List(Of List(Of Double))

                For i As Integer = 0 To rows - 1
                    Dim row As New List(Of Double)()

                    For j As Integer = 0 To columns - 1
                        row.Add(GetRandomValue)
                    Next

                    matrix.Add(row)
                Next

                Return matrix
            End Function

            Public Shared Function GetRandomValue() As Double
                ' Use your desired logic for weight initialization
                Dim rnd = New Random()
                Return rnd.Next(0.0, 1.0)
            End Function
            Public Shared Function InitializeWeightMatrix(ByVal inputSize As Integer, ByVal outputSize As Integer) As List(Of List(Of Double))
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

            Public Function Forward(input As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim output As New List(Of List(Of Double))

                For Each inputData In input
                    Dim weightedSum As New List(Of Double)

                    For i As Integer = 0 To Weights.Count - 1
                        Dim weightRow As List(Of Double) = Weights(i)
                        Dim weightedInput As Double = 0.0

                        For j As Integer = 0 To inputData.Count - 1
                            weightedInput += weightRow(j) * inputData(j)
                        Next

                        weightedSum.Add(weightedInput + ibias(i))
                    Next

                    output.Add(weightedSum)
                Next

                Return output
            End Function



        End Class
        Public Class MultMath
            Public Shared Function ActivationDerivative(ByVal vector As List(Of Double)) As List(Of Double)
                Dim result As List(Of Double) = New List(Of Double)()
                For Each val As Double In vector
                    result.Add(1 - Math.Tanh(val) ^ 2)
                Next
                Return result
            End Function

            Public Shared Function ActivationFunction(ByVal vector As List(Of Double)) As List(Of Double)
                Dim result As List(Of Double) = New List(Of Double)()
                For Each val As Double In vector
                    result.Add(Math.Tanh(val))
                Next
                Return result
            End Function
            Public Shared Function Activate(inputs As List(Of Double), transferFunction As TransferFunctionType) As List(Of Double)
                Select Case transferFunction
                    Case TransferFunctionType.Identity
                        Return inputs
                    Case TransferFunctionType.SoftMax
                        Return MultMath.SoftMax(inputs)
                    Case TransferFunctionType.Relu
                        Return MultMath.ReLU(inputs)
                    Case TransferFunctionType.Sigmoid
                        Return MultMath.Sigmoid(inputs)
                    Case Else
                        Return inputs
                End Select
            End Function

            Public Shared Function Activate(value As Double, transferFunction As TransferFunctionType) As Double
                Select Case transferFunction
                    Case TransferFunctionType.Sigmoid
                        Return 1.0 / (1.0 + Math.Exp(-value))
                ' Add other transfer functions as needed
                    Case TransferFunctionType.Relu
                        Return ReLU(value)
                    Case TransferFunctionType.BinaryThreshold
                        Return BinaryThreshold(value)


                    Case TransferFunctionType.HyperbolicTangent
                        Return Math.Tanh(value)
                    Case TransferFunctionType.BinaryThreshold
                        Return If(value >= 0, 1, 0)
                    Case TransferFunctionType.RectifiedLinear
                        Return Math.Max(0, value)
                    Case TransferFunctionType.Logistic
                        Return 1 / (1 + Math.Exp(-value))
                    Case TransferFunctionType.StochasticBinary
                        Return If(value >= 0, 1, 0)
                    Case TransferFunctionType.Gaussian
                        Return Math.Exp(-(value * value))
                    Case TransferFunctionType.Signum
                        Return Math.Sign(value)
                    Case Else
                        Throw New ArgumentException("Invalid transfer function type.")

                        Return value
                End Select
            End Function

            Public Shared Function AddVectors(ByVal vector1 As List(Of List(Of Double)), ByVal vector2 As List(Of List(Of Double))) As List(Of List(Of Double))
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

            ''' <summary>
            ''' the step function rarely performs well except in some rare cases with (0,1)-encoded
            ''' binary data.
            ''' </summary>
            ''' <param name="Value"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Public Shared Function BinaryThreshold(ByRef Value As Double) As Double

                ' Z = Bias+ (Input*Weight)
                'TransferFunction
                'If Z > 0 then Y = 1
                'If Z < 0 then y = 0

                Return If(Value < 0 = True, 0, 1)
            End Function

            Public Shared Function ComputeCrossEntropyLoss(predictions As List(Of List(Of Double)), targets As List(Of List(Of Double))) As Double
                ' Ensure predictions and targets have the same shape
                If predictions.Count <> targets.Count OrElse predictions(0).Count <> targets(0).Count Then
                    Throw New ArgumentException("Predictions and targets must have the same shape.")
                End If

                ' Compute the element-wise negative log likelihood
                Dim elementWiseLoss As New List(Of List(Of Double))()
                For i As Integer = 0 To predictions.Count - 1
                    Dim lossRow As New List(Of Double)()
                    For j As Integer = 0 To predictions(i).Count - 1
                        Dim p As Double = predictions(i)(j)
                        Dim t As Double = targets(i)(j)
                        Dim lossValue As Double = If(p > 0, -Math.Log(p) * t, 0)
                        lossRow.Add(lossValue)
                    Next
                    elementWiseLoss.Add(lossRow)
                Next

                ' Sum the losses across all elements and take the mean
                Dim totalLoss As Double = 0
                Dim numElements As Integer = 0
                For Each row In elementWiseLoss
                    For Each lossValue In row
                        totalLoss += lossValue
                        numElements += 1
                    Next
                Next
                Dim averageLoss As Double = totalLoss / numElements

                Return averageLoss
            End Function

            Public Shared Function ComputeCrossEntropyLossSoftmax(predictions As List(Of Double), targets As List(Of Double)) As Double
                ' Ensure predictions and targets have the same length
                If predictions.Count <> targets.Count Then
                    Throw New ArgumentException("Predictions and targets must have the same length.")
                End If

                Dim loss As Double = 0.0

                For i As Integer = 0 To predictions.Count - 1
                    loss += targets(i) * Math.Log(predictions(i))
                Next

                Return -loss
            End Function

            Public Shared Function ComputeGradients(predictions As List(Of List(Of Double)), targets As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim gradients As New List(Of List(Of Double))

                ' Iterate over each prediction and target pair
                For i As Integer = 0 To predictions.Count - 1
                    Dim prediction As List(Of Double) = predictions(i)
                    Dim target As List(Of Double) = targets(i)

                    Dim gradient As New List(Of Double)

                    ' Calculate gradient for each element in the prediction
                    For j As Integer = 0 To prediction.Count - 1
                        ' Compute the gradient of the cross-entropy loss with respect to each prediction element
                        Dim grad As Double = prediction(j) - target(j)
                        gradient.Add(grad)
                    Next

                    gradients.Add(gradient)
                Next

                Return gradients
            End Function

            Public Shared Function ComputeMean(sequence As List(Of Double)) As Double
                Dim sum = 0.0
                For Each value In sequence
                    sum += value
                Next

                Return sum / sequence.Count
            End Function

            Public Shared Function ComputeVariance(sequence As List(Of Double), mean As Double) As Double
                Dim sum = 0.0
                For Each value In sequence
                    sum += (value - mean) * (value - mean)
                Next

                Return sum / sequence.Count
            End Function

            Public Shared Function CreateRandomMatrix(rows As Integer, columns As Integer) As List(Of List(Of Double))
                Dim random As New Random()
                Dim matrix As New List(Of List(Of Double))

                For i As Integer = 0 To rows - 1
                    Dim row As New List(Of Double)()

                    For j As Integer = 0 To columns - 1
                        row.Add(MultMath.GetRandomValue)
                    Next

                    matrix.Add(row)
                Next

                Return matrix
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

            Public Shared Function DotProduct(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of List(Of Double))) As List(Of Double)
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

            Public Shared Function DotProduct(matrix1 As List(Of List(Of Double)), matrix2 As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim result As New List(Of List(Of Double))

                For i = 0 To matrix1.Count - 1
                    Dim row As New List(Of Double)

                    For j = 0 To matrix2(0).Count - 1
                        Dim sum As Double = 0.0

                        For k = 0 To matrix1(0).Count - 1
                            sum += matrix1(i)(k) * matrix2(k)(j)
                        Next

                        row.Add(sum)
                    Next

                    result.Add(row)
                Next

                Return result
            End Function

            Public Shared Function Gaussian(ByRef x As Double) As Double
                Gaussian = Math.Exp((-x * -x) / 2)
            End Function

            Public Shared Function GaussianDerivative(ByRef x As Double) As Double
                GaussianDerivative = Gaussian(x) * (-x / (-x * -x))
            End Function

            Public Shared Function GetRandomValue() As Double
                ' Use your desired logic for weight initialization
                Dim rnd = New Random()
                Return rnd.Next(0.0, 1.0)
            End Function

            Public Shared Function HyperbolicTangent(ByRef Value As Double) As Double
                ' TanH(x) = (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))

                Return Math.Tanh(Value)
            End Function

            Public Shared Function HyperbolicTangentDerivative(ByRef Value As Double) As Double
                HyperbolicTangentDerivative = 1 - (HyperbolicTangent(Value) * HyperbolicTangent(Value)) * Value
            End Function

            Public Shared Function MultiplyVectorByScalar(ByVal vector As List(Of Double), ByVal scalar As List(Of Double)) As List(Of Double)
                Dim result As List(Of Double) = New List(Of Double)()

                For i As Integer = 0 To vector.Count - 1
                    result.Add(vector(i) * scalar(i))
                Next

                Return result
            End Function

            Public Shared Function Normalize(ByRef Outputs As List(Of Double))
                Dim maxOutput As Double = Outputs.Max()
                Dim minOutput As Double = Outputs.Min()

                For i As Integer = 0 To Outputs.Count - 1
                    Outputs(i) = (Outputs(i) - minOutput) / (maxOutput - minOutput)
                Next
                Return Outputs
            End Function

            Public Shared Function OuterProduct(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of Double)) As List(Of List(Of Double))
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

            Public Shared Function ReLU(input As Double) As Double
                Return Math.Max(0.0, input)
            End Function

            Public Shared Function ReLU(inputs As List(Of Double)) As List(Of Double)
                Dim outputs = New List(Of Double)

                For i = 0 To inputs.Count - 1
                    outputs.Add(Math.Max(0.0, inputs(i)))
                Next

                Return outputs
            End Function

            Public Shared Function ScaleMatrix(matrix As List(Of List(Of Double)), scaleFactor As Integer) As List(Of List(Of Double))
                Dim result As New List(Of List(Of Double))

                For i = 0 To matrix.Count - 1
                    Dim newRow As New List(Of Double)
                    For j = 0 To matrix(i).Count - 1
                        newRow.Add(matrix(i)(j) / scaleFactor)
                    Next
                    result.Add(newRow)
                Next

                Return result
            End Function

            Public Shared Function ShuffleArrayToList(array As Integer()) As List(Of Integer)
                Dim random As New Random()
                Dim n As Integer = array.Length

                While n > 1
                    n -= 1
                    Dim k As Integer = random.Next(n + 1)
                    Dim value As Integer = array(k)
                    array(k) = array(n)
                    array(n) = value
                End While
                Return array.ToList
            End Function
            Public Shared Function Sigmoid(value As Double) As Double
                Return 1 / (1 + Math.Exp(-value))
            End Function

            Public Shared Function Sigmoid(inputs As List(Of Double)) As List(Of Double)
                Dim sigmoidOutputs As List(Of Double) = New List(Of Double)

                For Each value In inputs
                    sigmoidOutputs.Add(Sigmoid(value))
                Next

                Return sigmoidOutputs
            End Function

            Public Shared Function Softmax(matrix As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim result As New List(Of List(Of Double))

                For i = 0 To matrix.Count - 1
                    Dim row As New List(Of Double)
                    Dim maxValue As Double = matrix(i).Max()
                    Dim expSum As Double = 0.0

                    For j = 0 To matrix(i).Count - 1
                        expSum += Math.Exp(matrix(i)(j) - maxValue)
                    Next

                    For j = 0 To matrix(i).Count - 1
                        row.Add(Math.Exp(matrix(i)(j) - maxValue) / expSum)
                    Next

                    result.Add(row)
                Next

                Return result
            End Function

            Public Shared Function SoftMax(layer As List(Of Double)) As List(Of Double)
                Dim softmaxOutputs As List(Of Double) = New List(Of Double)
                Dim expSum As Double = 0.0

                For Each value In layer
                    expSum += Math.Exp(value)
                Next

                For Each value In layer
                    softmaxOutputs.Add(Math.Exp(value) / expSum)
                Next

                Return softmaxOutputs
            End Function

            Public Shared Function SubtractVectors(ByVal vector1 As List(Of Double), ByVal vector2 As List(Of Double)) As List(Of Double)
                Dim result As List(Of Double) = New List(Of Double)()

                For i As Integer = 0 To vector1.Count - 1
                    result.Add(vector1(i) - vector2(i))
                Next

                Return result
            End Function

            Public Shared Function TransposeMatrix(matrix As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim rows As Integer = matrix.Count
                Dim cols As Integer = matrix(0).Count

                Dim result As New List(Of List(Of Double))
                For i = 0 To cols - 1
                    Dim newRow As New List(Of Double)
                    For j = 0 To rows - 1
                        newRow.Add(matrix(j)(i))
                    Next
                    result.Add(newRow)
                Next

                Return result
            End Function

            ''' <summary>
            ''' Transposes vector from list of vectors . [1,2,3,4 ] to [[1],[2],[3],[4]]] rotating it
            ''' </summary>
            ''' <param name="vector"></param>
            ''' <returns></returns>
            Public Shared Function TransposeVector(vector As List(Of Double)) As List(Of List(Of Double))
                Dim result As New List(Of List(Of Double))

                For Each item In vector
                    Dim Lst As New List(Of Double)
                    Lst.Add(item)
                    result.Add(Lst)
                Next

                Return result
            End Function

            ''' <summary>
            ''' This does not produce a matrix it reduces the values according to the the trillinear mask
            ''' </summary>
            ''' <param name="matrix"></param>
            ''' <returns></returns>
            Public Shared Function TrillReductionMask(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
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
        End Class
        Public Class Trainer
            Inherits SingleLayerNeuralNetwork

            Public Sub New(ByRef PreTrained As SingleLayerNeuralNetwork)
                MyBase.New(PreTrained)
            End Sub
            Public Function Train(ByRef Inputs As List(Of Double), ByRef Targets As List(Of Double), iLearningRate As Double, Activation As TransferFunctionType) As Double
                Me.learningRate = iLearningRate
                '  Me.maxIterations = maxIterations

                ' Forward pass
                Forward(Inputs, Activation)

                ' Backward pass (calculate error and update weights)
                Dim outputErrors As List(Of Double) = CalculateOutputErrors(Targets, Activation)
                Dim hiddenErrors As List(Of Double) = CalculateHiddenErrors(outputErrors)

                UpdateWeights(Inputs, HiddenLayer, learningRate, hiddenErrors, outputErrors)

                ' Calculate total error
                Return CalculateTotalError(outputErrors)
            End Function
            Public Sub TrainRNN(ByVal inputSequence As List(Of Double), ByVal targetOutput As List(Of Double))
                Dim iteration As Integer = 0
                Dim ierror As Double = Double.PositiveInfinity

                While iteration < maxIterations AndAlso ierror > 0.001
                    ' Forward propagation
                    Dim hiddenLayerOutput As List(Of Double) = CalculateLayerOutput(inputSequence, InputWeights)
                    Dim outputLayerOutput As List(Of Double) = CalculateLayerOutput(hiddenLayerOutput, HiddenWeights)

                    ' Backpropagation
                    Dim outputError As List(Of Double) = MultMath.SubtractVectors(targetOutput, outputLayerOutput)
                    Dim outputDelta As List(Of Double) = MultMath.MultiplyVectorByScalar(outputError, MultMath.ActivationDerivative(outputLayerOutput))

                    Dim hiddenError As List(Of Double) = MultMath.DotProduct(outputDelta, MultMath.TransposeMatrix(HiddenWeights))
                    Dim hiddenDelta As List(Of Double) = MultMath.MultiplyVectorByScalar(hiddenError, MultMath.ActivationDerivative(hiddenLayerOutput))

                    ' Update weights
                    HiddenWeights = MultMath.AddVectors(HiddenWeights, MultMath.OuterProduct(hiddenLayerOutput, outputDelta))
                    InputWeights = MultMath.AddVectors(InputWeights, MultMath.OuterProduct(inputSequence, hiddenDelta))

                    ' Calculate total error
                    ierror = CalculateTotalError(targetOutput, outputLayerOutput)

                    iteration += 1
                End While
            End Sub
            ''' <summary>
            '''     Sigmoid Training:
            '''     The training Function Using the sigmoid output Is often used For binary classification problems Or
            '''     When Each output represents an independent probability.
            '''     The sigmoid Function squashes the output values between 0 And 1, representing the probability Of a binary Event.
            '''     This Function calculates the mean squared Error Or cross-entropy Error And updates the weights accordingly.
            ''' </summary>
            ''' <param name="Inputs"></param>
            ''' <param name="Targets"></param>
            ''' <param name="iLearningRate"></param>
            ''' <returns></returns>
            Public Function TrainSigmoid(ByRef Inputs As List(Of Double), ByRef Targets As List(Of Double), iLearningRate As Double) As Double
                Me.learningRate = iLearningRate
                '  Me.maxIterations = maxIterations
                ' Forward pass
                Forward(Inputs, TransferFunctionType.Sigmoid)

                ' Backward pass (calculate error and update weights)
                Dim outputErrors As List(Of Double) = CalculateOutputErrorsSigmoid(Targets)
                Dim hiddenErrors As List(Of Double) = CalculateHiddenErrors(outputErrors)

                UpdateWeights(Inputs, HiddenLayer, learningRate, hiddenErrors, outputErrors)

                ' Calculate total error
                Return CalculateTotalError(outputErrors)
            End Function
            Public Function TrainSoftMax(inputs As List(Of Double), targets As List(Of Double), ilearningRate As Double) As Double
                Me.learningRate = ilearningRate
                '  Me.maxIterations = maxIterations

                ' Perform forward pass
                Dim hiddenLayer As List(Of Double) = ForwardHidden(inputs, TransferFunctionType.Sigmoid)
                Dim outputLayer As List(Of Double) = ForwardOutput(hiddenLayer, TransferFunctionType.SoftMax)

                ' Calculate the cross-entropy loss
                'Dim loss As Double = ComputeCrossEntropyLossSoftmax(Outputs, targets)

                Dim loss As Double = 0.0
                For i As Integer = 0 To Me.outputNodes - 1
                    loss -= targets(i) * Math.Log(outputLayer(i)) + (1 - targets(i)) * Math.Log(1 - outputLayer(i))
                Next

                ' Perform backward pass and update weights
                Dim outputErrors As List(Of Double) = New List(Of Double)()
                For i As Integer = 0 To Me.outputNodes - 1
                    outputErrors.Add(outputLayer(i) - targets(i))
                Next

                Dim hiddenErrors As List(Of Double) = New List(Of Double)()
                For i As Integer = 0 To Me.hiddenNodes - 1
                    Dim ierror As Double = 0.0
                    For j As Integer = 0 To Me.outputNodes - 1
                        ierror += outputErrors(j) * Me.HiddenWeights(j)(i)
                    Next
                    hiddenErrors.Add(ierror)
                Next

                For i As Integer = 0 To Me.outputNodes - 1
                    For j As Integer = 0 To Me.hiddenNodes - 1
                        Me.HiddenWeights(i)(j) += learningRate * outputErrors(i) * hiddenLayer(j)
                    Next
                Next

                For i As Integer = 0 To Me.hiddenNodes - 1
                    For j As Integer = 0 To Me.inputNodes - 1
                        Me.InputWeights(i)(j) += learningRate * hiddenErrors(i) * inputs(j)
                    Next
                Next

                Return loss
            End Function
            Private Function CalculateOutputErrors(targets As List(Of Double), activation As TransferFunctionType) As List(Of Double)
                Dim outputErrors As List(Of Double) = New List(Of Double)
                Dim derivative As List(Of Double) = MultMath.Activate(Outputs, activation)

                For i As Integer = 0 To targets.Count - 1
                    Dim ierror As Double = targets(i) - Outputs(i)
                    outputErrors.Add(ierror * derivative(i))
                Next

                Return outputErrors
            End Function

            Private Function CalculateOutputErrorsSigmoid(targets As List(Of Double)) As List(Of Double)
                Dim outputErrors As List(Of Double) = New List(Of Double)
                Dim sigmoidDerivative As List(Of Double) = MultMath.Activate(Outputs, TransferFunctionType.Sigmoid_Derivative)

                For i As Integer = 0 To targets.Count - 1
                    Dim ierror As Double = (targets(i) - Outputs(i)) * sigmoidDerivative(i)
                    outputErrors.Add(ierror)
                Next

                Return outputErrors
            End Function
            Private Function CalculateOutputErrorsSoftmax(targets As List(Of Double)) As List(Of Double)
                Dim outputErrors As List(Of Double) = New List(Of Double)
                Dim softmaxOutputs As List(Of Double) = MultMath.Activate(Outputs, TransferFunctionType.SoftMax)

                For i As Integer = 0 To targets.Count - 1
                    Dim ierror As Double = softmaxOutputs(i) - targets(i)
                    outputErrors.Add(ierror)
                Next

                Return outputErrors
            End Function
        End Class
    End Class
End Namespace