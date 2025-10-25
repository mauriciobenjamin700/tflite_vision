/// Biblioteca de Base para o pacote `tflite_vision`.
/// Contém A classe [TFLiteModel] usada para modelos de segmentação e classificação TFLite.
/// Classes:
/// - TFLiteModel: Classe base para modelos TFLite, incluindo carregamento, gerenciamento de recursos e criação de buffers de saída.
library;

import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Classe que encapsula a funcionalidade de um modelo TFLite, incluindo
/// carregamento, gerenciamento de recursos e criação de buffers de saída.
/// 
/// Métodos:
/// - loadModel: Carrega o modelo TFLite e os rótulos a partir dos caminhos fornecidos.
/// - dispose: Libera os recursos associados ao modelo.
/// - validateOutputShapes: Valida as formas das saídas do modelo.
/// - createOutputBuffers: Cria buffers de saída com base nas formas das saídas do modelo.
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

  /// Carrega o modelo TFLite e os rótulos a partir dos caminhos fornecidos.
  /// Retorna uma instância de [TFLiteModel] pronta para uso.
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

  /// Libera os recursos associados ao modelo.
  /// Chame este método quando o modelo não for mais necessário.
  void dispose() {
    _isolateInterpreter.close();
    _mainInterpreter.close();
    debugPrint('Recursos do TFLiteModel liberados');
  }

  /// Valida as formas das saídas do modelo.
  /// Lança uma exceção se o número de saídas for menor que [expectedOutputs].
  void validateOutputShapes(List<List<int>> outputShapes, int expectedOutputs) {
    if (outputShapes.length < expectedOutputs) {
      throw Exception(
        'O modelo carregado não possui as $expectedOutputs saídas esperadas. '
        'Saídas: $outputShapes',
      );
    }
  }

  /// Cria buffers de saída com base nas formas das saídas do modelo.
  /// Retorna um mapa onde a chave é o índice da saída e o valor é o buffer correspondente.
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

  /// Getters
  Interpreter get mainInterpreter => _mainInterpreter;
  IsolateInterpreter get isolateInterpreter => _isolateInterpreter;
  List<int> get inputShape => _inputShape;
  List<List<int>> get outputShapes => _outputShapes;
  List<String> get labels => _labels;
}