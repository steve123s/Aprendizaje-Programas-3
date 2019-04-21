//
//  ViewController.swift
//  BackPropagacion-1
//
//  Created by Daniel Salinas on 4/20/19.
//  Copyright © 2019 DanielSteven. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        var trainingData: [[Float]] = []
        var trainingResults: [[Float]] = []
        // Fills both matrixs
        for p in -2...2 {
            let result = 1 + sin((Double.pi/2)*(Double(p)))
            trainingData.append([Float(p)])
            trainingResults.append([Float(result)])
        }
        
        print("trainingData: \(trainingData)")
        print("trainingResults: \(trainingResults)")
        
        // AND Trainig data
        // let traningDat: [[Float]] = [ [0,0], [0,1], [1,0], [1,1] ]
        // let traningResult: [[Float]] = [ [0], [0], [0], [1] ]
        
        let backProb = NeuralNetwork(inputSize: 1, hiddenSize: 2, outputSize: 1)
        
        for _ in 0..<NeuralNetwork.iterations {
            
            for i in 0..<trainingResults.count {
                backProb.train(input: trainingData[i], targetOutput: trainingResults[i], learningRate: NeuralNetwork.learningRate, momentum: NeuralNetwork.momentum)
            }
            
            for i in 0..<trainingResults.count {
                let t = trainingData[i]
                print("Valor esperado: \(trainingResults[i][0]) \t Valor obtenido: \(backProb.run(input: t)[0])")
                //Used for AND
                //print("\(t[0]), \(t[1])  -- \(backProb.run(input: t)[0])")
            }
            
        }
        
        
        
    }
    
    
}


public extension ClosedRange where Bound: FloatingPoint {
    public func random() -> Bound {
        let range = self.upperBound - self.lowerBound
        let randomValue = (Bound(arc4random_uniform(UINT32_MAX)) / Bound(UINT32_MAX)) * range + self.lowerBound
        return randomValue
    }
}

public class Layer {
    
    private var output: [Float]
    private var input: [Float]
    private var weights: [Float]
    private var previousWeights: [Float]
    
    init(inputSize: Int, outputSize: Int) {
        self.output = [Float](repeating: 0, count: outputSize)
        self.input = [Float](repeating: 0, count: inputSize + 1)
        self.weights = (0..<(1 + inputSize) * outputSize).map { _ in
            return (-0.5...0.5).random()
        }
        print(weights)
        previousWeights = [Float](repeating: 0, count: weights.count)
    }
    
    public func run(inputArray: [Float]) -> [Float] {
        
        for i in 0..<inputArray.count {
            input[i] = inputArray[i]
        }
        
        input[input.count-1] = 1
        var offSet = 0
        
        for i in 0..<output.count {
            for j in 0..<input.count {
                output[i] += weights[offSet+j] * input[j]
            }
            
            output[i] = ActivationFunction.sigmoid(x: output[i])
            offSet += input.count
            
        }
        
        return output
    }
    
    public func train(error: [Float], learningRate: Float, momentum: Float) -> [Float] {
        
        var offset = 0
        var nextError = [Float](repeating: 0, count: input.count)
        
        for i in 0..<output.count {
            
            let delta = error[i] * ActivationFunction.sigmoidDerivative(x: output[i])
            
            for j in 0..<input.count {
                let weightIndex = offset + j
                nextError[j] = nextError[j] + weights[weightIndex] * delta
                let dw = input[j] * delta * learningRate
                weights[weightIndex] += previousWeights[weightIndex] * momentum + dw
                previousWeights[weightIndex] = dw
            }
            
            offset += input.count
            
        }
        
        return nextError
    }
    
}

import Foundation

public class ActivationFunction {
    
    static func sigmoid(x: Float) -> Float {
        return 1 / (1 + exp(-x))
    }
    
    static func sigmoidDerivative(x: Float) -> Float {
        return x * (1 - x)
    }
    
}

import Foundation

public class NeuralNetwork {
    
    public static var learningRate: Float = 0.001
    public static var momentum: Float = 0.5
    public static var iterations: Int = 50000
    
    private var layers: [Layer] = []
    
    public init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        self.layers.append(Layer(inputSize: inputSize, outputSize: hiddenSize))
        self.layers.append(Layer(inputSize: hiddenSize, outputSize: outputSize))
    }
    
    public func run(input: [Float]) -> [Float] {
        
        var activations = input
        for i in 0..<layers.count {
            activations = layers[i].run(inputArray: activations)
        }
        
        return activations
    }
    
    public func train(input: [Float], targetOutput: [Float], learningRate: Float, momentum: Float) {
        
        let calculatedOutput = run(input: input)
        print("Output: \(calculatedOutput)")
        var error = zip(targetOutput, calculatedOutput).map { $0 - $1 }
        print("Error: \(error)")
        
        for i in (0...layers.count-1).reversed() {
            error = layers[i].train(error: error, learningRate: learningRate, momentum: momentum)
        }
    }
    
}
