import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

class PreProcessing {
  static Future<Uint8List> fromUiImage(ui.Image image) async {
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) {
      throw Exception('Falha ao converter imagem para bytes');
    }
    return byteData.buffer.asUint8List();
  }

  static Future<List<List<List<List<double>>>>> imageToTensor(
    Uint8List imageBytes,
    int tensorWidth,
    int tensorHeight,
  ) async {
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      throw Exception('Erro ao decodificar imagem');
    }

    final resizedImage = img.copyResize(
      originalImage,
      width: tensorWidth,
      height: tensorHeight,
    );

    final tensor = List.generate(
      1,
      (_) => List.generate(
        tensorHeight,
        (_) => List.generate(tensorWidth, (_) => List.generate(3, (_) => 0.0)),
      ),
    );

    int pixelIndex = 0;
    for (final pixel in resizedImage) {
      final y = pixelIndex ~/ tensorWidth;
      final x = pixelIndex % tensorWidth;

      tensor[0][y][x][0] = pixel.r / pixel.maxChannelValue;
      tensor[0][y][x][1] = pixel.g / pixel.maxChannelValue;
      tensor[0][y][x][2] = pixel.b / pixel.maxChannelValue;

      pixelIndex++;
    }

    return tensor;
  }

  /// Converte tensor normalizado (4D) de volta para imagem
  static img.Image tensorToImage(
    List<List<List<List<double>>>> tensor,
  ) {
    if (tensor.isEmpty || tensor[0].isEmpty || tensor[0][0].isEmpty) {
      throw Exception('Tensor vazio ou inválido');
    }

    final batchSize = tensor.length;
    final height = tensor[0].length;
    final width = tensor[0][0].length;
    final channels = tensor[0][0][0].length;

    if (batchSize != 1) {
      throw Exception('Tensor deve ter batch size = 1');
    }

    if (channels < 3) {
      throw Exception('Tensor deve ter pelo menos 3 canais (RGB)');
    }

    // Criar imagem com as dimensões do tensor
    final image = img.Image(width: width, height: height);

    // Converter cada pixel do tensor para a imagem
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Obter valores normalizados (0.0 - 1.0)
        final r = tensor[0][y][x][0];
        final g = tensor[0][y][x][1];
        final b = tensor[0][y][x][2];

        // Converter de volta para valores de 0-255
        final rValue = (r * 255).clamp(0, 255).round();
        final gValue = (g * 255).clamp(0, 255).round();
        final bValue = (b * 255).clamp(0, 255).round();

        // Definir o pixel na imagem
        image.setPixel(x, y, img.ColorRgb8(rValue, gValue, bValue));
      }
    }

    return image;
  }

  static Uint8List tensorToImageBytes(
    List<List<List<List<double>>>> tensor,
  ) {
    final image = tensorToImage(tensor);
    return Uint8List.fromList(img.encodePng(image));
  }

  /// Função que orquestra o pré-processamento completo
  static Future<List<List<List<List<double>>>>> processImage(
    Uint8List imageBytes, {
    int width = 640,
    int height = 640,
  }) async {
    try {
      final tensor = await imageToTensor(imageBytes, width, height);

      return tensor;
    } catch (error) {
      rethrow;
    }
  }
}
