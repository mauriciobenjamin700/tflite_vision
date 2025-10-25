import 'dart:typed_data';
import 'package:flutter/material.dart';
import './models.dart';
import './pre_processing.dart';
import './results.dart';

class AppState extends ChangeNotifier {
  TFLiteSegmentModel? _segmentationModel;
  TFLiteClassifyModel? _classifyModel;
  ClassificationResult? _classificationResult;
  SegmentationResult? _segmentationResult;
  Uint8List? _originalImage;
  Uint8List? _processedImage;
  bool _isLoading = false;
  bool _areModelsLoaded = false;

  // Paths configuráveis (padrões mantidos para compatibilidade)
  String segmentModelPath;
  String segmentLabelsPath;
  String classifyModelPath;
  String classifyLabelsPath;

  TFLiteSegmentModel? get segmentationModel => _segmentationModel;
  TFLiteClassifyModel? get classifyModel => _classifyModel;
  ClassificationResult? get classificationResult => _classificationResult;
  SegmentationResult? get segmentationResult => _segmentationResult;
  Uint8List? get originalImage => _originalImage;
  Uint8List? get processedImage => _processedImage;
  bool get isLoading => _isLoading;
  bool get areModelsLoaded => _areModelsLoaded;

  AppState({
    this.segmentModelPath = 'assets/segment_float16.tflite',
    this.segmentLabelsPath = 'assets/segment_labels.txt',
    this.classifyModelPath = 'assets/classify_float16.tflite',
    this.classifyLabelsPath = 'assets/classify_labels.txt',
  }) {
    _loadModels();
  }

  @override
  void dispose() {
    _segmentationModel?.dispose();
    _classifyModel?.dispose();
    super.dispose();
  }

  Future<void> _loadModels() async {
    try {
      _segmentationModel = await TFLiteSegmentModel.loadModel(
        modelPath: 'assets/segment_float16.tflite',
        labelsPath: 'assets/segment_labels.txt',
      );
      _classifyModel = await TFLiteClassifyModel.loadModel(
        modelPath: 'assets/classify_float16.tflite',
        labelsPath: 'assets/classify_labels.txt',
      );
      _areModelsLoaded = true;
    } catch (error) {
      _areModelsLoaded = false;
      throw Exception(error);
    }
    notifyListeners();
  }

  Future<void> selectImage(Uint8List imageBytes) async {
    _originalImage = imageBytes;
    _processedImage = null;
    _classificationResult = null;
    _segmentationResult = null;
    notifyListeners();
  }

  Future<void> runInference() async {
    if (_originalImage == null ||
        !_areModelsLoaded ||
        _segmentationModel == null ||
        _classifyModel == null) {
      return;
    }

    _isLoading = true;
    notifyListeners();

    try {
      // Segmentação
      final segmentationInputShape = _segmentationModel!.inputShape;
      final segmentationTensor = await PreProcessing.processImage(
        _originalImage!,
        width: segmentationInputShape[2],
        height: segmentationInputShape[1],
      );

      _segmentationResult = await _segmentationModel!.predict(
        segmentationTensor,
        _originalImage!,
      );

      _processedImage = _segmentationResult!.segmentedImage;
      notifyListeners();

      // Classificação usando a imagem segmentada
      final classificationInputShape = _classifyModel!.inputShape;
      final classificationTensor = await PreProcessing.processImage(
        _segmentationResult!.segmentedImage!,
        width: classificationInputShape[2],
        height: classificationInputShape[1],
      );

      _classificationResult = await _classifyModel!.predict(
        classificationTensor,
        _originalImage!,
      );
    } catch (error) {
      throw Exception(error);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Atualiza os paths dos modelos e recarrega-os imediatamente.
  Future<void> updateModelPaths({
    required String newSegmentModelPath,
    required String newSegmentLabelsPath,
    required String newClassifyModelPath,
    required String newClassifyLabelsPath,
  }) async {
    segmentModelPath = newSegmentModelPath;
    segmentLabelsPath = newSegmentLabelsPath;
    classifyModelPath = newClassifyModelPath;
    classifyLabelsPath = newClassifyLabelsPath;

    _areModelsLoaded = false;
    notifyListeners();
    await _loadModels();
  }

}
