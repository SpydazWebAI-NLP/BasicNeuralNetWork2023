Public Class NeuralNetwork_A
    Private inputLayerSize As Integer
    Public hiddenLayerSize As Integer
    Public outputLayerSize As Integer
    Public transferFunc As TransferFunction
    Public weightsIH As Double(,)
    Public weightsHO As Double(,)
    Private learningRate As Double
    ' Define the TransferFunction enum to represent different activation functions.
    Public Enum TransferFunction
        Sigmoid
        ReLU
        Tanh
    End Enum
    ' Constructor
    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double)
        Me.inputLayerSize = inputSize
        Me.hiddenLayerSize = hiddenSize
        Me.outputLayerSize = outputSize
        Me.transferFunc = transferFunc
        Me.learningRate = learningRate

        ' Initialize weights with random values (you can use other initialization methods)
        InitializeWeights()
    End Sub

    ' Private method to initialize weights with random values.
    Public Sub InitializeWeights()
        Dim random As New Random()

        ' Initialize weightsIH with random values between -1 and 1.
        weightsIH = New Double(inputLayerSize, hiddenLayerSize) {}
        For i As Integer = 0 To inputLayerSize - 1
            For j As Integer = 0 To hiddenLayerSize - 1
                weightsIH(i, j) = 2 * random.NextDouble() - 1
            Next
        Next

        ' Initialize weightsHO with random values between -1 and 1.
        weightsHO = New Double(hiddenLayerSize, outputLayerSize) {}
        For i As Integer = 0 To hiddenLayerSize - 1
            For j As Integer = 0 To outputLayerSize - 1
                weightsHO(i, j) = 2 * random.NextDouble() - 1
            Next
        Next
    End Sub

    ' Private method for forward propagation (predicting output given input).
    Public Function Forward(inputData As Double()) As Double()
        ' Calculate activations of hidden layer neurons.
        Dim hiddenLayerActivations As Double() = MatrixDotProduct(inputData, weightsIH)
        ApplyTransferFunction(hiddenLayerActivations)

        ' Calculate activations of output layer neurons.
        Dim outputLayerActivations As Double() = MatrixDotProduct(hiddenLayerActivations, weightsHO)
        ApplyTransferFunction(outputLayerActivations)

        Return outputLayerActivations
    End Function

    ' Private method to perform backpropagation and update weights during training.
    Private Sub TrainOneSample(inputData As Double(), targetData As Double())
        ' Perform forward propagation to calculate activations.
        Dim hiddenLayerActivations As Double() = MatrixDotProduct(inputData, weightsIH)
        ApplyTransferFunction(hiddenLayerActivations)

        Dim outputLayerActivations As Double() = MatrixDotProduct(hiddenLayerActivations, weightsHO)
        ApplyTransferFunction(outputLayerActivations)

        ' Calculate output layer errors.
        Dim outputLayerErrors As Double() = CalculateOutputErrors(targetData, outputLayerActivations)

        ' Calculate hidden layer errors.
        Dim hiddenLayerErrors As Double() = CalculateHiddenErrors(outputLayerErrors, weightsHO)

        ' Update weights using SGD optimization.
        UpdateWeightsSGD(inputData, hiddenLayerActivations, outputLayerErrors, hiddenLayerErrors)
    End Sub

    ' Private method to calculate output layer errors.
    Public Function CalculateOutputErrors(targetData As Double(), outputActivations As Double()) As Double()
        Dim errors As Double() = New Double(outputLayerSize) {}
        For i As Integer = 0 To outputLayerSize - 1
            errors(i) = targetData(i) - outputActivations(i)
        Next
        Return errors
    End Function

    ' Private method to calculate hidden layer errors.
    Public Function CalculateHiddenErrors(outputErrors As Double(), weightsHO As Double(,)) As Double()
        ' Calculate hidden layer errors using backpropagation.
        Dim hiddenLayerErrors As Double() = New Double(hiddenLayerSize) {}

        For i As Integer = 0 To hiddenLayerSize - 1
            For j As Integer = 0 To outputLayerSize - 1
                hiddenLayerErrors(i) += outputErrors(j) * weightsHO(i, j)
            Next
        Next

        Return hiddenLayerErrors
    End Function

    ' Private method to update weights using SGD optimization.
    Public Sub UpdateWeightsSGD(inputData As Double(), hiddenLayerActivations As Double(), outputErrors As Double(), hiddenErrors As Double())
        ' Update weights using Stochastic Gradient Descent (SGD) optimization.
        For i As Integer = 0 To hiddenLayerSize - 1
            For j As Integer = 0 To outputLayerSize - 1
                weightsHO(i, j) += learningRate * outputErrors(j) * hiddenLayerActivations(i)
            Next
        Next

        For i As Integer = 0 To inputLayerSize - 1
            For j As Integer = 0 To hiddenLayerSize - 1
                weightsIH(i, j) += learningRate * hiddenErrors(j) * inputData(i)
            Next
        Next
    End Sub

    ' ... (existing code)

    ' Helper method to apply the transfer function to a matrix.
    Public Sub ApplyTransferFunction(ByRef matrix As Double())
        ' Apply the specified transfer function (e.g., Sigmoid, ReLU, Tanh) to the elements of the matrix.
        For i As Integer = 0 To matrix.Length - 1
            Select Case transferFunc
                Case TransferFunction.Sigmoid
                    matrix(i) = Sigmoid(matrix(i))
                Case TransferFunction.ReLU
                    matrix(i) = ReLU(matrix(i))
                Case TransferFunction.Tanh
                    matrix(i) = Tanh(matrix(i))
                    ' Add more transfer functions if needed.
            End Select
        Next
    End Sub

    ' Sigmoid activation function.
    Public Function Sigmoid(x As Double) As Double
        Return 1.0 / (1.0 + Math.Exp(-x))
    End Function

    ' ReLU activation function.
    Public Function ReLU(x As Double) As Double
        Return Math.Max(0, x)
    End Function

    ' Tanh activation function.
    Public Function Tanh(x As Double) As Double
        Return Math.Tanh(x)
    End Function

    ' Helper method for matrix dot product.
    Public Function MatrixDotProduct(matrixA As Double(), matrixB As Double(,)) As Double()
        ' Perform matrix dot product between matrixA and matrixB.
        Dim result As Double() = New Double(matrixB.GetLength(1) - 1) {}
        For i As Integer = 0 To matrixA.Length - 1
            For j As Integer = 0 To matrixB.GetLength(1) - 1
                result(j) += matrixA(i) * matrixB(i, j)
            Next
        Next
        Return result
    End Function
    ' Private method to update the network weights using the entire training dataset.
    Public Sub TrainFullDataset(trainingData As List(Of Tuple(Of Double(), Double())))
        For Each sample In trainingData
            TrainOneSample(sample.Item1, sample.Item2)
        Next
    End Sub



    ' Helper method for cloning the current neural network.
    Public Function Clone() As NeuralNetwork_A
        ' Create a deep copy of the current neural network.
        Dim clonedNN As New NeuralNetwork_A(inputLayerSize, hiddenLayerSize, outputLayerSize, transferFunc, learningRate)
        clonedNN.weightsIH = CType(weightsIH.Clone(), Double(,))
        clonedNN.weightsHO = CType(weightsHO.Clone(), Double(,))
        Return clonedNN
    End Function
End Class
' Define the RecurrentNeuralNetwork class inheriting from the NeuralNetwork class.
Public Class RecurrentNeuralNetwork
    Inherits NeuralNetwork_A

    ' Additional member for RNN: Number of time steps (sequence length).
    Private timeSteps As Integer

    ' Constructor to initialize RNN-specific parameters.
    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, timeSteps As Integer)
        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
        Me.timeSteps = timeSteps
    End Sub

    ' Private method for forward propagation in RNN (predicting output given input sequence).
    Public Function ForwardSequence(sequence As List(Of Double())) As List(Of Double())
        ' Perform forward propagation for the entire sequence.
        Dim outputSequence As New List(Of Double())()

        ' Initial hidden state for the first time step.
        Dim hiddenState As Double() = New Double(hiddenLayerSize) {}

        ' Loop through each time step in the sequence.
        For Each inputData In sequence
            ' Calculate activations of hidden layer neurons.
            Dim hiddenLayerActivations As Double() = MatrixDotProduct(inputData, weightsIH)
            ApplyTransferFunction(hiddenLayerActivations)

            ' Calculate activations of output layer neurons.
            Dim outputLayerActivations As Double() = MatrixDotProduct(hiddenLayerActivations, weightsHO)
            ApplyTransferFunction(outputLayerActivations)

            ' Update the hidden state for the next time step.
            hiddenState = hiddenLayerActivations

            ' Add the output activations of the current time step to the output sequence.
            outputSequence.Add(outputLayerActivations)
        Next

        Return outputSequence
    End Function

    ' Private method to perform backpropagation and update weights during training for a sequence.
    Public Sub TrainSequence(sequence As List(Of Double()), targetSequence As List(Of Double()))
        ' Perform forward propagation to calculate activations for the entire sequence.
        Dim outputSequence As List(Of Double()) = ForwardSequence(sequence)

        ' Initialize the overall errors for each time step.
        Dim overallOutputErrors(outputLayerSize) As Double

        ' Loop through each time step in reverse to calculate errors and update weights.
        For i As Integer = timeSteps - 1 To 0 Step -1
            Dim outputErrors As Double() = CalculateOutputErrors(targetSequence(i), outputSequence(i))
            overallOutputErrors = MatrixAdd(overallOutputErrors, outputErrors)

            ' Calculate hidden layer errors.
            Dim hiddenLayerErrors As Double() = CalculateHiddenErrors(overallOutputErrors, weightsHO)
            Dim hiddenLayerActivations As Double() = MatrixDotProduct(sequence(i), weightsIH)
            ' Update weights using SGD optimization.
            UpdateWeightsSGD(sequence(i), hiddenLayerActivations, overallOutputErrors, hiddenLayerErrors)
        Next
    End Sub

    ' Overriding the Train method for RNN training.
    Public Overloads Function Train(trainingData As List(Of Tuple(Of List(Of Double()), List(Of Double())))) As RecurrentNeuralNetwork
        ' Create a copy of the current RNN to train.
        Dim trainedRNN As RecurrentNeuralNetwork = CType(Me.MemberwiseClone(), RecurrentNeuralNetwork)

        ' Train the copy with the provided training sequences.
        For Each sequenceData In trainingData
            trainedRNN.TrainSequence(sequenceData.Item1, sequenceData.Item2)
        Next

        ' Return the trained RNN.
        Return trainedRNN
    End Function

    ' ... (other existing code)

    ' Helper method for matrix addition.
    Private Function MatrixAdd(matrixA As Double(), matrixB As Double()) As Double()
        ' Perform element-wise matrix addition between matrixA and matrixB.
        Dim result As Double() = New Double(matrixA.Length - 1) {}
        For i As Integer = 0 To matrixA.Length - 1
            result(i) = matrixA(i) + matrixB(i)
        Next
        Return result
    End Function
End Class
' Define the interface class to create and train neural network models.
Public Class NeuralNetworkInterface

    Inherits NeuralNetwork_A

    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double)
        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
    End Sub
    ''' <summary>
    ''' Trains the Transformer decoder network using the tokenized and indexed training data.
    ''' </summary>
    ''' <param name="trainingData">A list of input sequences (tokenized and indexed).</param>
    ''' <param name="labels">A list of target sequences (tokenized and indexed).</param>
    ''' <param name="epochs">The number of training epochs.</param>
    Public Sub TrainDecoder(trainingData As List(Of List(Of Integer)), labels As List(Of List(Of Integer)), epochs As Integer)
        ' Implement the training process using the given training data and labels.
        ' Apply backpropagation and stochastic gradient descent (SGD) optimization.
        ' Use the specified number of epochs for training.
    End Sub
    ''' <summary>
    ''' Trains the Transformer encoder network using the tokenized and indexed training data.
    ''' </summary>
    ''' <param name="trainingData">A list of input sequences (tokenized and indexed).</param>
    ''' <param name="labels">A list of target sequences (tokenized and indexed).</param>
    ''' <param name="epochs">The number of training epochs.</param>
    Public Sub TrainEncoder(trainingData As List(Of List(Of Integer)), labels As List(Of List(Of Integer)), epochs As Integer)
        ' Implement the training process using the given training data and labels.
        ' Apply backpropagation and stochastic gradient descent (SGD) optimization.
        ' Use the specified number of epochs for training.
    End Sub
    ''' <summary>
    ''' Trains the Transformer encoder-decoder network using the tokenized and indexed training data.
    ''' </summary>
    ''' <param name="trainingData">A list of input sequences (tokenized and indexed).</param>
    ''' <param name="labels">A list of target sequences (tokenized and indexed).</param>
    ''' <param name="epochs">The number of training epochs.</param>
    Public Sub TrainEncoderDecoder(trainingData As List(Of List(Of Integer)), labels As List(Of List(Of Integer)), epochs As Integer)
        ' Implement the training process using the given training data and labels.
        ' Apply backpropagation and stochastic gradient descent (SGD) optimization.
        ' Use the specified number of epochs for training.
    End Sub
    ' Public method to train the network using SGD optimization.
    Public Function Train(trainingData As List(Of Tuple(Of Double(), Double()))) As NeuralNetwork_A
        ' Code to create a copy of the current network to train.
        Dim trainedNetwork As NeuralNetwork_A = Me.Clone()

        ' Train the copy with the provided training data.
        trainedNetwork.TrainFullDataset(trainingData)

        ' Return the trained network.
        Return trainedNetwork
    End Function

    ' ... (existing code)
End Class
' Define the SoftmaxLayer class.
Public Class SoftmaxLayer
    ' Private member for the number of output neurons.
    Private outputSize As Integer

    ' Constructor to initialize SoftmaxLayer specific parameters.
    Public Sub New(outputSize As Integer)
        Me.outputSize = outputSize
    End Sub

    ' Public method for forward propagation with softmax output.
    Public Function Forward(inputData As Double()) As Double()
        ' Calculate activations of output layer neurons using softmax function.
        Dim outputLayerActivations As Double() = Softmax(inputData)
        Return outputLayerActivations
    End Function

    ' Private method for softmax function.
    Private Function Softmax(inputData As Double()) As Double()
        ' Compute the softmax function to get the output probabilities.
        Dim maxVal As Double = inputData.Max()
        Dim expScores As Double() = New Double(outputSize - 1) {}
        Dim sumExpScores As Double = 0.0

        For i As Integer = 0 To outputSize - 1
            expScores(i) = Math.Exp(inputData(i) - maxVal)
            sumExpScores += expScores(i)
        Next

        For i As Integer = 0 To outputSize - 1
            expScores(i) /= sumExpScores
        Next

        Return expScores
    End Function
End Class
Public Class NextWordPrediction
    Public Shared Sub Run()
        ' Sample training data for language modeling (a collection of sentences).
        Dim trainingData As List(Of String) = New List(Of String) From
            {
                "The quick brown",
                "She sells sea",
                "To be or",
                "A stitch in time",
                "All that glitters is",
                "Where there is smoke, there is"
            }

        ' Preprocess the training data and create vocabulary and token sequences.
        Dim vocabulary As HashSet(Of String) = New HashSet(Of String)()
        Dim tokenizedSentences As List(Of List(Of String)) = New List(Of List(Of String))()

        For Each sentence As String In trainingData
            Dim tokens() As String = sentence.Split(" "c)
            For Each token As String In tokens
                vocabulary.Add(token)
            Next
            tokenizedSentences.Add(tokens.ToList())
        Next

        ' Create a word-to-index mapping for the vocabulary.
        Dim wordToIndex As Dictionary(Of String, Integer) = vocabulary.Select(Function(word, index) (word, index)).ToDictionary(Function(tuple) tuple.word, Function(tuple) tuple.index)

        ' Convert token sequences into index sequences using the word-to-index mapping.
        Dim indexSequences As List(Of List(Of Integer)) = tokenizedSentences.Select(Function(tokens) tokens.Select(Function(token) wordToIndex(token)).ToList()).ToList()

        ' Define the vocabulary size and maximum sequence length.
        Dim vocabSize As Integer = vocabulary.Count
        Dim sequenceLength As Integer = tokenizedSentences.Max(Function(tokens) tokens.Count)

        ' Create a recurrent neural network for next-word prediction with ReLU transfer function.
        Dim languageModel As New RecurrentNeuralNetwork(inputSize:=vocabSize, hiddenSize:=50, outputSize:=vocabSize, transferFunc:=NeuralNetwork_A.TransferFunction.ReLU, learningRate:=0.01, timeSteps:=sequenceLength)

        ' Train the language model using the tokenized and indexed training data.
        ' ...

        ' Sample input sequence for prediction (the beginning of a sentence).
        Dim inputSequence As List(Of Double) = New List(Of Double) From {wordToIndex("The"), wordToIndex("quick")}

        ' Generate the next word predictions using the trained language model.
        Dim nextWordPredictions As Double() = languageModel.Forward(inputSequence.ToArray)

        ' Find the index of the word with the highest predicted probability.
        Dim nextWordIndex As Integer = Array.IndexOf(nextWordPredictions, nextWordPredictions.Max())

        ' Convert the predicted index back to the actual word using the word-to-index mapping.
        Dim predictedNextWord As String = wordToIndex.FirstOrDefault(Function(entry) entry.Value = nextWordIndex).Key

        ' Display the next-word prediction for the input sequence.
        Console.WriteLine($"Input Sequence: The quick")
        Console.WriteLine($"Predicted Next Word: {predictedNextWord}")
    End Sub
End Class





Namespace NOGOOD
    '' Define the TransformerEncoderDecoderNetwork class inheriting from the NeuralNetwork class.
    'Public Class TransformerEncoderDecoderNetwork
    '    Inherits NeuralNetwork

    '    ' Private member for the number of self-attention heads in the encoder.
    '    Private encoderSelfAttentionHeads As Integer

    '    ' Private member for the number of cross-attention heads in the decoder.
    '    Private decoderCrossAttentionHeads As Integer

    '    ' Private member for the size of the feed-forward layers.
    '    Private feedForwardSize As Integer

    '    ' Constructor to initialize TransformerEncoderDecoderNetwork specific parameters.
    '    Public Sub New(encoderInputSize As Integer, decoderInputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, encoderSelfAttentionHeads As Integer, decoderCrossAttentionHeads As Integer, feedForwardSize As Integer)
    '        MyBase.New(encoderInputSize, hiddenSize, outputSize, transferFunc, learningRate)
    '        Me.encoderSelfAttentionHeads = encoderSelfAttentionHeads
    '        Me.decoderCrossAttentionHeads = decoderCrossAttentionHeads
    '        Me.feedForwardSize = feedForwardSize
    '        Me.decoderInputSize = decoderInputSize
    '    End Sub

    '    ' Override the BuildModel method to create the Transformer encoder-decoder architecture.
    '    Protected Overrides Sub BuildModel()
    '        ' Add the self-attention layer in the encoder.
    '        Me.AddLayer(New SelfAttentionLayer(inputSize:=Me.hiddenSize, attentionHeads:=Me.encoderSelfAttentionHeads))

    '        ' Add the cross-attention layer in the decoder.
    '        Me.AddLayer(New CrossAttentionLayer(inputSize:=Me.hiddenSize, attentionHeads:=Me.decoderCrossAttentionHeads))

    '        ' Add a feed-forward layer with ReLU activation in the encoder.
    '        Me.AddLayer(New FeedForwardLayer(inputSize:=Me.hiddenSize, hiddenSize:=Me.feedForwardSize, activationFunc:=TransferFunction.ReLU))

    '        ' Add another feed-forward layer with ReLU activation in the encoder.
    '        Me.AddLayer(New FeedForwardLayer(inputSize:=Me.feedForwardSize, hiddenSize:=Me.hiddenSize, activationFunc:=TransferFunction.ReLU))

    '        ' Add the output layer with softmax activation in the decoder.
    '        Me.AddLayer(New SoftmaxLayer(outputSize:=Me.outputSize))
    '    End Sub
    'End Class
    'Public Class MachineTranslation
    '    Public Shared Sub Run()
    '        ' Sample training data for machine translation (English to French).
    '        Dim englishSentences As List(Of String) = New List(Of String) From
    '        {
    '            "The quick brown fox jumps over the lazy dog.",
    '            "She sells sea shells by the sea shore.",
    '            "To be or not to be, that is the question.",
    '            "A stitch in time saves nine.",
    '            "All that glitters is not gold.",
    '            "Where there is smoke, there is fire."
    '        }

    '        Dim frenchSentences As List(Of String) = New List(Of String) From
    '        {
    '            "Le renard brun rapide saute par-dessus le chien paresseux.",
    '            "Elle vend des coquillages au bord de la mer.",
    '            "Être ou ne pas être, telle est la question.",
    '            "Un point à temps en sauve neuf.",
    '            "Tout ce qui brille n'est pas de l'or.",
    '            "Où il y a de la fumée, il y a du feu."
    '        }

    '        ' Preprocess the training data and create vocabulary and token sequences for both English and French.
    '        ' ...

    '        ' Define the vocabulary sizes for both English and French.
    '        Dim englishVocabSize As Integer = englishVocabulary.Count
    '        Dim frenchVocabSize As Integer = frenchVocabulary.Count

    '        ' Define the maximum sequence lengths for both English and French.
    '        Dim englishMaxSequenceLength As Integer = englishTokenizedSentences.Max(Function(tokens) tokens.Count)
    '        Dim frenchMaxSequenceLength As Integer = frenchTokenizedSentences.Max(Function(tokens) tokens.Count)

    '        ' Create a Transformer encoder network for machine translation.
    '        Dim transformerEncoder As New TransformerEncoderNetwork(inputSize:=englishVocabSize, hiddenSize:=256, outputSize:=frenchVocabSize, transferFunc:=TransferFunction.ReLU, learningRate:=0.01, attentionHeads:=8, feedForwardSize:=512)

    '        ' Train the Transformer encoder network using the tokenized and indexed training data.
    '        ' ...

    '        ' Sample input sequence for translation (English sentence).
    '        Dim englishInputSequence As List(Of Integer) = New List(Of Integer) From {englishWordToIndex("The"), englishWordToIndex("quick"), englishWordToIndex("brown"), englishWordToIndex("fox")}

    '        ' Generate the translation using the trained Transformer encoder.
    '        Dim frenchTranslationIndices As List(Of Integer) = transformerEncoder.Forward(englishInputSequence.ToArray()).ToList()

    '        ' Convert the predicted French indices back to the actual words using the French index-to-word mapping.
    '        Dim frenchTranslation As List(Of String) = frenchTranslationIndices.Select(Function(index) frenchIndexToWord(index)).ToList()

    '        ' Display the translation for the input English sentence.
    '        Console.WriteLine($"Input Sentence: The quick brown fox")
    '        Console.WriteLine($"Translated Sentence: {String.Join(" ", frenchTranslation)}")
    '    End Sub
    'End Class
    '' Define a use case for these models: Image Classification.
    'Public Class ImageClassification
    '    Public Shared Sub Run()
    '        ' Sample image data (features).
    '        Dim imageData As Double() = New Double() {0.5, 0.8, 0.3, 0.1, 0.9}

    '        ' Create a neural network for image classification using Sigmoid transfer function.
    '        Dim imageClassifier As New NeuralNetwork(inputSize:=5, hiddenSize:=10, outputSize:=3, transferFunc:=NeuralNetwork.TransferFunction.Sigmoid, learningRate:=0.01)

    '        ' Train the neural network with some labeled data (features and targets).
    '        ' ...

    '        ' Perform image classification using the trained neural network.
    '        Dim predictedProbabilities As Double() = imageClassifier.Forward(imageData)

    '        ' Display the predicted probabilities for each class.
    '        Console.WriteLine("Predicted Probabilities:")
    '        For i As Integer = 0 To predictedProbabilities.Length - 1
    '            Console.WriteLine($"Class {i + 1}: {predictedProbabilities(i)}")
    '        Next

    '        ' Create a recurrent neural network for sequential image classification using ReLU transfer function.
    '        Dim sequentialImageClassifier As New RecurrentNeuralNetwork(inputSize:=5, hiddenSize:=10, outputSize:=3, transferFunc:=NeuralNetwork.TransferFunction.ReLU, learningRate:=0.01, timeSteps:=4)

    '        ' Train the recurrent neural network with sequential data.
    '        ' ...

    '        ' Perform sequential image classification using the trained recurrent neural network.
    '        Dim sequentialImageData As List(Of Double()) = New List(Of Double()) From
    '            {
    '                New Double() {0.5, 0.8, 0.3, 0.1, 0.9},
    '                New Double() {0.3, 0.6, 0.2, 0.5, 0.7},
    '                New Double() {0.2, 0.7, 0.4, 0.6, 0.8},
    '                New Double() {0.1, 0.5, 0.3, 0.7, 0.6}
    '            }

    '        Dim predictedProbabilitiesSeq As List(Of Double()) = sequentialImageClassifier.ForwardSequence(sequentialImageData)

    '        ' Display the predicted probabilities for each class at each time step.
    '        Console.WriteLine("Predicted Probabilities (Sequential):")
    '        For t As Integer = 0 To predictedProbabilitiesSeq.Count - 1
    '            Console.WriteLine($"Time Step {t + 1}:")
    '            For i As Integer = 0 To predictedProbabilitiesSeq(t).Length - 1
    '                Console.WriteLine($"Class {i + 1}: {predictedProbabilitiesSeq(t)(i)}")
    '            Next
    '        Next

    '        ' Create a self-attention neural network for image classification using Tanh transfer function.
    '        Dim selfAttentionImageClassifier As New SelfAttentionNeuralNetwork(inputSize:=5, hiddenSize:=10, outputSize:=3, transferFunc:=NeuralNetwork.TransferFunction.Tanh, learningRate:=0.01, attentionHeads:=2)

    '        ' Train the self-attention neural network with attention mechanism.
    '        ' ...

    '        ' Perform image classification using the trained self-attention neural network.
    '        Dim predictedProbabilitiesAttention As Double() = selfAttentionImageClassifier.Forward(imageData)

    '        ' Display the predicted probabilities for each class with attention mechanism.
    '        Console.WriteLine("Predicted Probabilities (Attention):")
    '        For i As Integer = 0 To predictedProbabilitiesAttention.Length - 1
    '            Console.WriteLine($"Class {i + 1}: {predictedProbabilitiesAttention(i)}")
    '        Next

    '        ' Create a masked self-attention neural network for sequential image classification using ReLU transfer function.
    '        Dim maskedSelfAttentionSequentialImageClassifier As New MaskedSelfAttentionNeuralNetwork(inputSize:=5, hiddenSize:=10, outputSize:=3, transferFunc:=NeuralNetwork.TransferFunction.ReLU, learningRate:=0.01, attentionHeads:=2)

    '        ' Train the masked self-attention neural network with masked attention mechanism.
    '        ' ...

    '        ' Perform sequential image classification using the trained masked self-attention neural network.
    '        Dim predictedProbabilitiesMaskedAttention As List(Of Double()) = maskedSelfAttentionSequentialImageClassifier.ForwardSequence(sequentialImageData)

    '        ' Display the predicted probabilities for each class at each time step with masked attention.
    '        Console.WriteLine("Predicted Probabilities (Masked Attention):")
    '        For t As Integer = 0 To predictedProbabilitiesMaskedAttention.Count - 1
    '            Console.WriteLine($"Time Step {t + 1}:")
    '            For i As Integer = 0 To predictedProbabilitiesMaskedAttention(t).Length - 1
    '                Console.WriteLine($"Class {i + 1}: {predictedProbabilitiesMaskedAttention(t)(i)}")
    '            Next
    '        Next

    '        ' Create a multi-head attention neural network for image classification using Sigmoid transfer function.
    '        Dim multiHeadAttentionImageClassifier As New MultiHeadAttentionNeuralNetwork(inputSize:=5, hiddenSize:=10, outputSize:=3, transferFunc:=NeuralNetwork.TransferFunction.Sigmoid, learningRate:=0.01, attentionHeads:=3, useMask:=True)

    '        ' Train the multi-head attention neural network with masked and multi-head attention mechanism.
    '        ' ...

    '        ' Perform image classification using the trained multi-head attention neural network.
    '        Dim predictedProbabilitiesMultiHead As Double() = multiHeadAttentionImageClassifier.Forward(imageData)

    '        ' Display the predicted probabilities for each class with multi-head attention and mask.
    '        Console.WriteLine("Predicted Probabilities (Multi-Head Attention with Mask):")
    '        For i As Integer = 0 To predictedProbabilitiesMultiHead.Length - 1
    '            Console.WriteLine($"Class {i + 1}: {predictedProbabilitiesMultiHead(i)}")
    '        Next
    '    End Sub
    'End Class
    '' Define the MultiHeadAttentionNeuralNetwork class inheriting from the NeuralNetwork class.
    'Public Class MultiHeadAttentionNeuralNetwork
    '    Inherits NeuralNetwork

    '    ' Additional members for Multi-Head Attention: Number of attention heads and the mask flag.
    '    Private attentionHeads As Integer
    '    Private useMask As Boolean

    '    ' Constructor to initialize Multi-Head Attention specific parameters.
    '    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, attentionHeads As Integer, useMask As Boolean)
    '        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
    '        Me.attentionHeads = attentionHeads
    '        Me.useMask = useMask
    '    End Sub

    '    ' Private method for Multi-Head Attention mechanism.
    '    Private Function MultiHeadAttention(inputData As Double()) As Double()
    '        ' Calculate activations of hidden layer neurons.
    '        Dim hiddenLayerActivations As Double() = MatrixDotProduct(inputData, weightsIH)
    '        ApplyTransferFunction(hiddenLayerActivations)

    '        ' Apply Multi-Head Attention mechanism to the hidden layer activations.
    '        hiddenLayerActivations = ApplyMultiHeadAttention(hiddenLayerActivations)

    '        ' Calculate activations of output layer neurons.
    '        Dim outputLayerActivations As Double() = MatrixDotProduct(hiddenLayerActivations, weightsHO)
    '        ApplyTransferFunction(outputLayerActivations)

    '        Return outputLayerActivations
    '    End Function

    '    ' Private method to apply Multi-Head Attention mechanism.
    '    Private Function ApplyMultiHeadAttention(hiddenLayerActivations As Double()) As Double()
    '        ' Split the hidden layer activations into multiple heads.
    '        Dim headSize As Integer = hiddenLayerSize \ attentionHeads
    '        Dim splitHeads(,) As Double = New Double(attentionHeads - 1, headSize - 1) {}

    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                splitHeads(head, i) = hiddenLayerActivations(head * headSize + i)
    '            Next
    '        Next

    '        ' Perform Multi-Head Attention within each attention head.
    '        Dim scaledAttentionOutputs(,) As Double = New Double(attentionHeads - 1, headSize - 1) {}
    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                If useMask Then
    '                    scaledAttentionOutputs(head, i) = PerformScaledDotProductMaskedAttention(splitHeads(head, i), splitHeads(head, 0 To headSize - 1), headSize, i)
    '                Else
    '                    scaledAttentionOutputs(head, i) = PerformScaledDotProductAttention(splitHeads(head, i), splitHeads(head, 0 To headSize - 1))
    '                End If
    '            Next
    '        Next

    '        ' Concatenate the attention outputs from all heads.
    '        Dim concatenatedAttentionOutputs As Double() = New Double(hiddenLayerSize - 1) {}
    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                concatenatedAttentionOutputs(head * headSize + i) = scaledAttentionOutputs(head, i)
    '            Next
    '        Next

    '        Return concatenatedAttentionOutputs
    '    End Function



    'End Class
    '' Define the SelfAttentionNeuralNetwork class inheriting from the NeuralNetwork class.
    'Public Class SelfAttentionNeuralNetwork
    '    Inherits NeuralNetwork

    '    ' Additional member for Self-Attention: Number of attention heads.
    '    Private attentionHeads As Integer

    '    ' Constructor to initialize Self-Attention specific parameters.
    '    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, attentionHeads As Integer)
    '        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
    '        Me.attentionHeads = attentionHeads
    '    End Sub

    '    ' Private method for Self-Attention mechanism.
    '    Private Function SelfAttention(inputData As Double()) As Double()
    '        ' Calculate activations of hidden layer neurons.
    '        Dim hiddenLayerActivations As Double() = MatrixDotProduct(inputData, weightsIH)
    '        ApplyTransferFunction(hiddenLayerActivations)

    '        ' Apply Self-Attention mechanism to the hidden layer activations.
    '        hiddenLayerActivations = ApplySelfAttention(hiddenLayerActivations)

    '        ' Calculate activations of output layer neurons.
    '        Dim outputLayerActivations As Double() = MatrixDotProduct(hiddenLayerActivations, weightsHO)
    '        ApplyTransferFunction(outputLayerActivations)

    '        Return outputLayerActivations
    '    End Function

    '    ' Private method to apply Self-Attention mechanism.
    '    Private Function ApplySelfAttention(hiddenLayerActivations As Double()) As Double()
    '        ' Split the hidden layer activations into multiple heads.
    '        Dim headSize As Integer = hiddenLayerSize \ attentionHeads
    '        Dim splitHeads(,) As Double = New Double(attentionHeads - 1, headSize - 1) {}

    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                splitHeads(head, i) = hiddenLayerActivations(head * headSize + i)
    '            Next
    '        Next

    '        ' Perform Self-Attention within each attention head.
    '        Dim scaledAttentionOutputs(,) As Double = New Double(attentionHeads - 1, headSize - 1) {}
    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                scaledAttentionOutputs(head, i) = PerformScaledDotProductAttention(splitHeads(head, i), splitHeads(head, 0 To headSize - 1))
    '            Next
    '        Next

    '        ' Concatenate the attention outputs from all heads.
    '        Dim concatenatedAttentionOutputs As Double() = New Double(hiddenLayerSize - 1) {}
    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                concatenatedAttentionOutputs(head * headSize + i) = scaledAttentionOutputs(head, i)
    '            Next
    '        Next

    '        Return concatenatedAttentionOutputs
    '    End Function

    '    ' Private method to perform Scaled Dot-Product Attention within a single attention head.
    '    Private Function PerformScaledDotProductAttention(query As Double(), keys() As Double) As Double
    '        ' Perform scaled dot-product attention.
    '        Dim dotProduct As Double = MatrixDotProduct(query, keys)
    '        Dim attentionScore As Double = dotProduct / Math.Sqrt(keys.Length)

    '        ' Apply softmax to the attention scores.
    '        Dim attentionWeights() As Double = Softmax(attentionScore, keys.Length)

    '        ' Calculate the weighted sum using attention weights.
    '        Dim weightedSum As Double = 0.0
    '        For i As Integer = 0 To keys.Length - 1
    '            weightedSum += attentionWeights(i) * keys(i)
    '        Next

    '        Return weightedSum
    '    End Function

    '    ' Helper method for softmax.
    '    Public Function Softmax(attentionScore As Double, keySize As Integer) As Double()
    '        ' Compute the softmax function to convert attention scores into attention weights.
    '        Dim attentionWeights() As Double = New Double(keySize - 1) {}
    '        Dim sumExpScores As Double = 0.0

    '        For i As Integer = 0 To keySize - 1
    '            attentionWeights(i) = Math.Exp(attentionScore)
    '            sumExpScores += attentionWeights(i)
    '        Next

    '        For i As Integer = 0 To keySize - 1
    '            attentionWeights(i) /= sumExpScores
    '        Next

    '        Return attentionWeights
    '    End Function



    'End Class
    '' Define the MaskedSelfAttentionNeuralNetwork class inheriting from the NeuralNetwork class.
    'Public Class MaskedSelfAttentionNeuralNetwork
    '    Inherits NeuralNetwork

    '    ' Additional member for Masked Self-Attention: Number of attention heads.
    '    Private attentionHeads As Integer

    '    ' Constructor to initialize Masked Self-Attention specific parameters.
    '    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, attentionHeads As Integer)
    '        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
    '        Me.attentionHeads = attentionHeads
    '    End Sub

    '    ' Private method for Masked Self-Attention mechanism.
    '    Private Function MaskedSelfAttention(inputData As Double()) As Double()
    '        ' Calculate activations of hidden layer neurons.
    '        Dim hiddenLayerActivations As Double() = MatrixDotProduct(inputData, weightsIH)
    '        ApplyTransferFunction(hiddenLayerActivations)

    '        ' Apply Masked Self-Attention mechanism to the hidden layer activations.
    '        hiddenLayerActivations = ApplyMaskedSelfAttention(hiddenLayerActivations)

    '        ' Calculate activations of output layer neurons.
    '        Dim outputLayerActivations As Double() = MatrixDotProduct(hiddenLayerActivations, weightsHO)
    '        ApplyTransferFunction(outputLayerActivations)

    '        Return outputLayerActivations
    '    End Function

    '    ' Private method to apply Masked Self-Attention mechanism.
    '    Private Function ApplyMaskedSelfAttention(hiddenLayerActivations As Double()) As Double()
    '        ' Split the hidden layer activations into multiple heads.
    '        Dim headSize As Integer = hiddenLayerSize \ attentionHeads
    '        Dim splitHeads(,) As Double = New Double(attentionHeads - 1, headSize - 1) {}

    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                splitHeads(head, i) = hiddenLayerActivations(head * headSize + i)
    '            Next
    '        Next

    '        ' Perform Masked Self-Attention within each attention head.
    '        Dim scaledAttentionOutputs(,) As Double = New Double(attentionHeads - 1, headSize - 1) {}
    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                scaledAttentionOutputs(head, i) = PerformScaledDotProductMaskedAttention(splitHeads(head, i), splitHeads(head, 0 To headSize - 1), headSize, i)
    '            Next
    '        Next

    '        ' Concatenate the attention outputs from all heads.
    '        Dim concatenatedAttentionOutputs As Double() = New Double(hiddenLayerSize - 1) {}
    '        For head As Integer = 0 To attentionHeads - 1
    '            For i As Integer = 0 To headSize - 1
    '                concatenatedAttentionOutputs(head * headSize + i) = scaledAttentionOutputs(head, i)
    '            Next
    '        Next

    '        Return concatenatedAttentionOutputs
    '    End Function

    '    ' Private method to perform Scaled Dot-Product Masked Attention within a single attention head.
    '    Private Function PerformScaledDotProductMaskedAttention(query As Double(,), keys(,) As Double, keySize As Integer, currentPos As Integer) As Double
    '        ' Perform scaled dot-product masked attention.
    '        Dim dotProduct As Double = MatrixDotProduct(query, keys)
    '        Dim attentionScore As Double = dotProduct / Math.Sqrt(keySize)

    '        ' Apply masking to the attention score.
    '        If currentPos < keySize Then
    '            attentionScore = Double.NegativeInfinity
    '        End If

    '        ' Apply softmax to the attention scores.
    '        Dim attentionWeights() As Double = Softmax(attentionScore, keys.Length)

    '        ' Calculate the weighted sum using attention weights.
    '        Dim weightedSum As Double = 0.0
    '        For i As Integer = 0 To keys.Length - 1
    '            weightedSum += attentionWeights(i) * keys(i)
    '        Next

    '        Return weightedSum
    '    End Function

    '    ' Helper method for softmax.
    '    Public Function Softmax(attentionScore As Double, keySize As Integer) As Double()
    '        ' Compute the softmax function to convert attention scores into attention weights.
    '        Dim attentionWeights() As Double = New Double(keySize - 1) {}
    '        Dim sumExpScores As Double = 0.0

    '        For i As Integer = 0 To keySize - 1
    '            attentionWeights(i) = Math.Exp(attentionScore)
    '            sumExpScores += attentionWeights(i)
    '        Next

    '        For i As Integer = 0 To keySize - 1
    '            attentionWeights(i) /= sumExpScores
    '        Next

    '        Return attentionWeights
    '    End Function
    'End Class
    '' Define the TransformerEncoderNetwork class inheriting from the NeuralNetwork class.
    'Public Class TransformerEncoderNetwork
    '    Inherits NeuralNetwork

    '    ' Private member for the number of attention heads in the self-attention mechanism.
    '    Private attentionHeads As Integer

    '    ' Private member for the size of the feed-forward layers.
    '    Private feedForwardSize As Integer

    '    ' Constructor to initialize TransformerEncoderNetwork specific parameters.
    '    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, attentionHeads As Integer, feedForwardSize As Integer)
    '        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
    '        Me.attentionHeads = attentionHeads
    '        Me.feedForwardSize = feedForwardSize
    '    End Sub

    '    ' Override the BuildModel method to create the Transformer encoder architecture.
    '    Protected Overrides Sub BuildModel()
    '        ' Add the self-attention layer.
    '        Me.AddLayer(New SelfAttentionLayer(inputSize:=Me.hiddenSize, attentionHeads:=Me.attentionHeads))

    '        ' Add a feed-forward layer with ReLU activation.
    '        Me.AddLayer(New FeedForwardLayer(inputSize:=Me.hiddenSize, hiddenSize:=Me.feedForwardSize, activationFunc:=TransferFunction.ReLU))

    '        ' Add another feed-forward layer with ReLU activation.
    '        Me.AddLayer(New FeedForwardLayer(inputSize:=Me.feedForwardSize, hiddenSize:=Me.hiddenSize, activationFunc:=TransferFunction.ReLU))

    '        ' Add the output layer with softmax activation.
    '        Me.AddLayer(New SoftmaxLayer(outputSize:=Me.outputSize))
    '    End Sub
    'End Class

    '' Define the TransformerDecoderNetwork class inheriting from the NeuralNetwork class.
    'Public Class TransformerDecoderNetwork
    '    Inherits NeuralNetwork

    '    ' Private member for the number of self-attention heads in the decoder.
    '    Private selfAttentionHeads As Integer

    '    ' Private member for the number of cross-attention heads in the decoder.
    '    Private crossAttentionHeads As Integer

    '    ' Private member for the size of the feed-forward layers.
    '    Private feedForwardSize As Integer

    '    ' Constructor to initialize TransformerDecoderNetwork specific parameters.
    '    Public Sub New(inputSize As Integer, hiddenSize As Integer, outputSize As Integer, transferFunc As TransferFunction, learningRate As Double, selfAttentionHeads As Integer, crossAttentionHeads As Integer, feedForwardSize As Integer)
    '        MyBase.New(inputSize, hiddenSize, outputSize, transferFunc, learningRate)
    '        Me.selfAttentionHeads = selfAttentionHeads
    '        Me.crossAttentionHeads = crossAttentionHeads
    '        Me.feedForwardSize = feedForwardSize
    '    End Sub

    '    ' Override the BuildModel method to create the Transformer decoder architecture.
    '    Protected Overrides Sub BuildModel()
    '        ' Add the self-attention layer.
    '        Me.AddLayer(New SelfAttentionLayer(inputSize:=Me.hiddenSize, attentionHeads:=Me.selfAttentionHeads))

    '        ' Add the cross-attention layer.
    '        Me.AddLayer(New CrossAttentionLayer(inputSize:=Me.hiddenSize, attentionHeads:=Me.crossAttentionHeads))

    '        ' Add a feed-forward layer with ReLU activation.
    '        Me.AddLayer(New FeedForwardLayer(inputSize:=Me.hiddenSize, hiddenSize:=Me.feedForwardSize, activationFunc:=TransferFunction.ReLU))

    '        ' Add another feed-forward layer with ReLU activation.
    '        Me.AddLayer(New FeedForwardLayer(inputSize:=Me.feedForwardSize, hiddenSize:=Me.hiddenSize, activationFunc:=TransferFunction.ReLU))

    '        ' Add the output layer with softmax activation.
    '        Me.AddLayer(New SoftmaxLayer(outputSize:=Me.outputSize))
    '    End Sub
    'End Class
    'Public Class MachineTranslationWithDecoder
    '    Public Shared Sub Run()
    '        ' Sample training data for machine translation (English to French).
    '        Dim englishSentences As List(Of String) = New List(Of String) From
    '        {
    '            "The quick brown fox jumps over the lazy dog.",
    '            "She sells sea shells by the sea shore.",
    '            "To be or not to be, that is the question.",
    '            "A stitch in time saves nine.",
    '            "All that glitters is not gold.",
    '            "Where there is smoke, there is fire."
    '        }

    '        Dim frenchSentences As List(Of String) = New List(Of String) From
    '        {
    '            "Le renard brun rapide saute par-dessus le chien paresseux.",
    '            "Elle vend des coquillages au bord de la mer.",
    '            "Être ou ne pas être, telle est la question.",
    '            "Un point à temps en sauve neuf.",
    '            "Tout ce qui brille n'est pas de l'or.",
    '            "Où il y a de la fumée, il y a du feu."
    '        }

    '        ' Preprocess the training data and create vocabulary and token sequences for both English and French.
    '        ' ...

    '        ' Define the vocabulary sizes for both English and French.
    '        Dim englishVocabSize As Integer = englishVocabulary.Count
    '        Dim frenchVocabSize As Integer = frenchVocabulary.Count

    '        ' Define the maximum sequence lengths for both English and French.
    '        Dim englishMaxSequenceLength As Integer = englishTokenizedSentences.Max(Function(tokens) tokens.Count)
    '        Dim frenchMaxSequenceLength As Integer = frenchTokenizedSentences.Max(Function(tokens) tokens.Count)

    '        ' Create a Transformer encoder network for machine translation.
    '        Dim transformerEncoder As New TransformerEncoderNetwork(inputSize:=englishVocabSize, hiddenSize:=256, outputSize:=frenchVocabSize, transferFunc:=TransferFunction.ReLU, learningRate:=0.01, attentionHeads:=8, feedForwardSize:=512)

    '        ' Train the Transformer encoder network using the tokenized and indexed training data.
    '        ' ...

    '        ' Create a Transformer decoder network for machine translation.
    '        Dim transformerDecoder As New TransformerDecoderNetwork(inputSize:=frenchVocabSize, hiddenSize:=256, outputSize:=englishVocabSize, transferFunc:=TransferFunction.ReLU, learningRate:=0.01, selfAttentionHeads:=8, crossAttentionHeads:=8, feedForwardSize:=512)

    '        ' Train the Transformer decoder network using the tokenized and indexed training data.
    '        ' ...

    '        ' Sample input sequence for translation (English sentence).
    '        Dim englishInputSequence As List(Of Integer) = New List(Of Integer) From {englishWordToIndex("The"), englishWordToIndex("quick"), englishWordToIndex("brown"), englishWordToIndex("fox")}

    '        ' Generate the translation using the trained Transformer encoder.
    '        Dim frenchTranslationIndices As List(Of Integer) = transformerEncoder.Forward(englishInputSequence.ToArray()).ToList()

    '        ' Generate the translation using the trained Transformer decoder.
    '        Dim decodedEnglishIndices As List(Of Integer) = transformerDecoder.Forward(frenchTranslationIndices.ToArray()).ToList()

    '        ' Convert the predicted English indices back to the actual words using the English index-to-word mapping.
    '        Dim decodedEnglishTranslation As List(Of String) = englishIndexToWord(decodedEnglishIndices)

    '        ' Display the translation for the input English sentence.
    '        Console.WriteLine($"Input Sentence: The quick brown fox")
    '        Console.WriteLine($"Decoded English Translation: {String.Join(" ", decodedEnglishTranslation)}")
    '    End Sub
    'End Class
    'Public Class MachineTranslationWithEncoderDecoder
    '    Public Shared Sub Run()
    '        ' Sample training data for machine translation (English to French).
    '        Dim englishSentences As List(Of String) = New List(Of String) From
    '        {
    '            "The quick brown fox jumps over the lazy dog.",
    '            "She sells sea shells by the sea shore.",
    '            "To be or not to be, that is the question.",
    '            "A stitch in time saves nine.",
    '            "All that glitters is not gold.",
    '            "Where there is smoke, there is fire."
    '        }

    '        Dim frenchSentences As List(Of String) = New List(Of String) From
    '        {
    '            "Le renard brun rapide saute par-dessus le chien paresseux.",
    '            "Elle vend des coquillages au bord de la mer.",
    '            "Être ou ne pas être, telle est la question.",
    '            "Un point à temps en sauve neuf.",
    '            "Tout ce qui brille n'est pas de l'or.",
    '            "Où il y a de la fumée, il y a du feu."
    '        }

    '        ' Preprocess the training data and create vocabulary and token sequences for both English and French.
    '        ' ...

    '        ' Define the vocabulary sizes for both English and French.
    '        Dim englishVocabSize As Integer = englishVocabulary.Count
    '        Dim frenchVocabSize As Integer = frenchVocabulary.Count

    '        ' Define the maximum sequence lengths for both English and French.
    '        Dim englishMaxSequenceLength As Integer = englishTokenizedSentences.Max(Function(tokens) tokens.Count)
    '        Dim frenchMaxSequenceLength As Integer = frenchTokenizedSentences.Max(Function(tokens) tokens.Count)

    '        ' Create a Transformer encoder-decoder network for machine translation.
    '        Dim transformerEncoderDecoder As New TransformerEncoderDecoderNetwork(encoderInputSize:=englishVocabSize, decoderInputSize:=frenchVocabSize, hiddenSize:=256, outputSize:=frenchVocabSize, transferFunc:=TransferFunction.ReLU, learningRate:=0.01, encoderSelfAttentionHeads:=8, decoderCrossAttentionHeads:=8, feedForwardSize:=512)

    '        ' Train the Transformer encoder-decoder network using the tokenized and indexed training data.
    '        ' ...

    '        ' Sample input sequence for translation (English sentence).
    '        Dim englishInputSequence As List(Of Integer) = New List(Of Integer) From {englishWordToIndex("The"), englishWordToIndex("quick"), englishWordToIndex("brown"), englishWordToIndex("fox")}

    '        ' Generate the translation using the trained Transformer encoder-decoder.
    '        Dim frenchTranslationIndices As List(Of Integer) = transformerEncoderDecoder.Forward(englishInputSequence.ToArray()).ToList()

    '        ' Convert the predicted French indices back to the actual words using the French index-to-word mapping.
    '        Dim frenchTranslation As List(Of String) = frenchIndexToWord(frenchTranslationIndices)

    '        ' Display the translation for the input English sentence.
    '        Console.WriteLine($"Input Sentence: The quick brown fox")
    '        Console.WriteLine($"Translated Sentence: {String.Join(" ", frenchTranslation)}")
    '    End Sub
    'End Class
End Namespace
