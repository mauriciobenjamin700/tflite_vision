import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteModel {
  final Interpreter _mainInterpreter;
  final IsolateInterpreter _isolateInterpreter;
  final List<int> _inputShape;
  final List<List<int>> _outputShapes;
  final List<String> _labels;

  TFLiteModel({
    required Interpreter mainInterpreter,
    required IsolateInterpreter isolateInterpreter,
    required List<int> inputShape,
    required List<List<int>> outputShapes,
    required List<String> labels,
  })  : _mainInterpreter = mainInterpreter,
        _isolateInterpreter = isolateInterpreter,
        _inputShape = inputShape,
        _outputShapes = outputShapes,
        _labels = labels;

  static Future<TFLiteModel> loadModel({
    required String modelPath,
    required String labelsPath,
  }) async {
    try {
      debugPrint('Carregando modelo de $modelPath...');
      final mainInterpreter = await Interpreter.fromAsset(modelPath);
      final isolateInterpreter =
          await IsolateInterpreter.create(address: mainInterpreter.address);

      final labelsData = await rootBundle.loadString(labelsPath);
      final labels = const LineSplitter().convert(labelsData);

      final inputShape = mainInterpreter.getInputTensor(0).shape;
      final outputShapes = <List<int>>[];
      for (int i = 0; i < mainInterpreter.getOutputTensors().length; i++) {
        outputShapes.add(mainInterpreter.getOutputTensor(i).shape);
      }

      debugPrint('Modelo carregado com sucesso!');
      debugPrint('Input shape: $inputShape');
      debugPrint('Output shapes: $outputShapes');
      debugPrint('Labels: $labels');

      return TFLiteModel(
          mainInterpreter: mainInterpreter,
          isolateInterpreter: isolateInterpreter,
          inputShape: inputShape,
          outputShapes: outputShapes,
          labels: labels);
    } catch (error) {
      debugPrint('Erro ao carregar modelo: $error');
      rethrow;
    }
  }

  void dispose() {
    _isolateInterpreter.close();
    _mainInterpreter.close();
    debugPrint('Recursos do TFLiteModel liberados');
  }

  void validateOutputShapes(List<List<int>> outputShapes, int expectedOutputs) {
    if (outputShapes.length < expectedOutputs) {
      throw Exception(
        'O modelo carregado não possui as $expectedOutputs saídas esperadas. '
        'Saídas: $outputShapes',
      );
    }
  }

Map<int, Object> createOutputBuffers(List<List<int>> outputShapes) {
  final outputs = <int, Object>{};

  for (int i = 0; i < outputShapes.length; i++) {
    final shape = outputShapes[i];

    if (shape.length == 4) {
      outputs[i] = List.generate(
        shape[0],
        (_) => List.generate(
          shape[1],
          (_) => List.generate(
            shape[2],
            (_) => List.filled(shape[3], 0.0),
          ),
        ),
      );
    } else if (shape.length == 3) {
      outputs[i] = List.generate(
        shape[0],
        (_) => List.generate(
          shape[1],
          (_) => List.filled(shape[2], 0.0),
        ),
      );
    } else if (shape.length == 2) {
      outputs[i] = List.generate(
        shape[0],
        (_) => List<double>.filled(shape[1], 0.0),
      );
    } else if (shape.length == 1) {
      outputs[i] = List<double>.filled(shape[0], 0.0);
    }
  }

  return outputs;
}


  Interpreter get mainInterpreter => _mainInterpreter;
  IsolateInterpreter get isolateInterpreter => _isolateInterpreter;
  List<int> get inputShape => _inputShape;
  List<List<int>> get outputShapes => _outputShapes;
  List<String> get labels => _labels;
}