# Sneaker Classification Project 👟

## 📌 Descripción
Clasificador de zapatillas implementado con Transfer Learning usando EfficientNetB0. El modelo reconoce 50 modelos diferentes de zapatillas a partir de imágenes. 

**Nota sobre la versión actual**: Este es un MVP enfocado en validar el enfoque técnico. El modelo actual:
- Logra alta precisión con imágenes del dataset de entrenamiento (~85% val_accuracy)
- Está optimizado para ejecutarse con recursos limitados
- Requerirá mejoras para generalizar mejor con imágenes del mundo real

## 🚀 Características Principales
- **Arquitectura**: EfficientNetB0 preentrenada + capas personalizadas
- **Preprocesamiento**: Imágenes redimensionadas a 224x224 píxeles
- **Clasificación**: 50 modelos de zapatillas de marcas populares
- **Rendimiento**: 85% de precisión en validación (dataset de prueba)
- **Eficiencia**: Inferencia en ~50ms (CPU)
