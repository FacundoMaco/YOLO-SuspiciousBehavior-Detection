# Cómo subir a un nuevo repositorio de GitHub

## Paso 1: Crear nuevo repositorio en GitHub

1. Ve a https://github.com/new
2. Crea un nuevo repositorio con el nombre que desees (ej: `YOLO-SuspiciousBehavior-Detection`)
3. **NO** inicialices con README, .gitignore o licencia (ya los tenemos)
4. Copia la URL del repositorio (ej: `https://github.com/tuusuario/nombre-repo.git`)

## Paso 2: Cambiar el remoto y subir

Ejecuta estos comandos reemplazando la URL con la de tu nuevo repositorio:

```bash
# Opción A: Cambiar el remoto existente
git remote set-url origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main

# Opción B: Renombrar el remoto actual y agregar nuevo
git remote rename origin old-origin
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

## Archivos que NO se subirán (por .gitignore)

- ✅ Modelos entrenados (*.pt) - se descargan automáticamente
- ✅ Datasets (dataset/, train/, valid/) - muy grandes para GitHub
- ✅ Resultados de entrenamiento (runs/)
- ✅ Archivos temporales y cache

## Nota sobre archivos grandes

Los modelos base (yolov8n.pt ~6MB, yolov8n-pose.pt ~12MB) están excluidos del .gitignore.
Si quieres incluirlos, puedes usar Git LFS para archivos grandes:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS for model files"
```

## Estado actual

✅ Código commitado y listo para subir
✅ .gitignore configurado correctamente
✅ README.md completo con documentación

Solo necesitas crear el repositorio en GitHub y cambiar el remoto.

