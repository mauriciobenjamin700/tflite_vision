import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

int getBestSegmentationIndex(List<List<double>> detections, double threshold) {
  double maxConfidence = -1.0;
  int bestDetectionIndex = -1;
  int detectionCount = detections[0].length;

  for (int i = 0; i < detectionCount; i++) {
    final confidence = detections[4][i];
    if (confidence > 0.5 && confidence > maxConfidence) {
      maxConfidence = confidence;
      bestDetectionIndex = i;
    }
  }

  return bestDetectionIndex;
}


List<double> extractMaskCoefficients(
  List<List<double>> detections,
  int detectionIndex,
  int firstCoeffIndex,
) {
  final maskCoeffs = <double>[];
  for (int i = firstCoeffIndex; i < detections.length; i++) {
    maskCoeffs.add(detections[i][detectionIndex]);
  }
  return maskCoeffs;
}


img.Image buildBinaryMask(
  List<List<List<double>>> maskPrototypes,
  List<double> maskCoefficients,
) {
  final height = maskPrototypes.length;
  final width = maskPrototypes[0].length;
  final channels = maskPrototypes[0][0].length;
  final numCoeffs = math.min(maskCoefficients.length, channels);

  final binaryMask = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double maskValue = 0.0;

      // eixo dos canais no final (y, x, i)
      for (int i = 0; i < numCoeffs; i++) {
        maskValue += maskCoefficients[i] * maskPrototypes[y][x][i];
      }

      // Aplicar função sigmoide
      final sigmoidValue = 1 / (1 + math.exp(-maskValue));

      // Binarizar com threshold 0.5
      final pixelValue = sigmoidValue > 0.5 ? 255 : 0;

      binaryMask.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
    }
  }

  return binaryMask;
}


img.Image decodeOriginalImage(Uint8List bytes) {
  final image = img.decodeImage(bytes);
  if (image == null) {
    throw Exception("Falha ao decodificar a imagem original.");
  }
  return image;
}


img.Image resizeMask(img.Image mask, int targetWidth, int targetHeight) {
  return img.copyResize(
    mask,
    width: targetWidth,
    height: targetHeight,
    interpolation: img.Interpolation.linear,
  );
}


img.Image applyMaskToImage(img.Image image, img.Image mask) {
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final maskPixel = mask.getPixel(x, y);
      // Canal R, G e B são iguais, então usamos R
      if (maskPixel.r < 128) {
        image.setPixelRgb(x, y, 0, 0, 0);
      }
    }
  }
  return image;
}


Uint8List encodeImageToPng(img.Image image) {
  return Uint8List.fromList(img.encodePng(image));
}



(int, double) getMaxIndexAndProb(List<double> probabilities) {
  double maxProb = 0.0;
  int maxIndex = -1;
  for (int i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }
  if (maxIndex == -1) {
    throw Exception('No classification result found.');
  }

  return (maxIndex, maxProb);
}



