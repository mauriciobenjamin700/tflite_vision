import 'dart:typed_data';
import 'package:image/image.dart' as img;


abstract class Result {
  final double confidence;
  final DateTime timestamp;
  final Uint8List originalImage;
  
  const Result({
    required this.confidence,
    required this.timestamp,
    required this.originalImage,
  });
}

class ClassificationResult extends Result {
  final String label;
  final List<double>? allProbabilities;
  
  const ClassificationResult({
    required super.confidence,
    required super.timestamp,
    required super.originalImage,
    required this.label,
    this.allProbabilities,
  });
  
  @override
  String toString() {
    return 'ClassificationResult(label: $label, confidence: $confidence)';
  }
}


class SegmentationResult extends Result {
  final Uint8List? segmentedImage;
  final img.Image? binaryMask;
  final List<double>? maskCoefficients;
  final int? bestSegmentationIndex;
  final Map<String, dynamic>? metadata;
  
  const SegmentationResult({
    required super.confidence,
    required super.timestamp,
    required super.originalImage,
    this.segmentedImage,
    this.binaryMask,
    this.maskCoefficients,
    this.bestSegmentationIndex,
    this.metadata,
  });
  
  @override
  String toString() {
    return 'SegmentationResult(confidence: $confidence, hasSegmentedImage: ${segmentedImage != null})';
  }
}