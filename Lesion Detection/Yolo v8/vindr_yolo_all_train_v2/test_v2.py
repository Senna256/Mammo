from pathlib import Path

from train_all_vindr_v2 import (
    DICOMTrainer,
    DICOMValidator,
    DATASET,
    IMG_SIZE,
    BATCH,
    WORKERS,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "/home/enric_sena/Desktop/Mammo/runs/detect/train-19/weights/best.pt"
)

TEST_IMAGES = DATASET / "images" / "test"

DEVICE = 0

SAVE_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/runs/detect/test_v2"
)

# V2 preprocessing
WINDOWING_METHOD = "breast_tissue"
CALC_WINDOW = True
VOI_FUNC = "LINEAR"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLOv8 - TEST FINAL VinDr-Mammo V2")
    print("=" * 60)

    print(f"Modelo:        {MODEL_PATH}")
    print(f"Test:          {TEST_IMAGES}")
    print(f"Device:        cuda:{DEVICE}")
    print(f"Workers:       {WORKERS}")
    print(f"Method:        {WINDOWING_METHOD}")
    print(f"Calc window:   {CALC_WINDOW}")
    print(f"VOI function:  {VOI_FUNC}")
    print()

    # --------------------------------------------------------
    # COMPROBACIONES
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo:\n{MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"No existe el directorio:\n{TEST_IMAGES}"
        )

    n_images = len(
        list(
            TEST_IMAGES.glob("*.jpg")
        )
    )

    print(
        f"[TEST] Imágenes encontradas: "
        f"{n_images}"
    )

    if n_images != 4000:
        raise RuntimeError(
            f"Esperaba 4000 imágenes de test, "
            f"pero hay {n_images}"
        )

    # --------------------------------------------------------
    # CREAR TRAINER
    # --------------------------------------------------------

    print()
    print(
        "[TEST] Creando DICOMTrainer..."
    )

    overrides = {

        "model": str(
            MODEL_PATH
        ),

        "data": str(
            DATASET / "data.yaml"
        ),

        "imgsz": IMG_SIZE,
        "batch": BATCH,

        "workers": WORKERS,
        "device": DEVICE,

        "task": "detect",
        "mode": "val",

        "split": "test",

        "single_cls": True,

        "plots": True,
        "save_json": True,
        "verbose": True,

        "project": str(
            SAVE_DIR.parent
        ),

        "name": SAVE_DIR.name,

        "exist_ok": True,

        "conf": 0.001,
        "iou": 0.7,
    }

    trainer = DICOMTrainer(
        overrides=overrides
    )

    # --------------------------------------------------------
    # CONFIGURAR PREPROCESSING V2
    # --------------------------------------------------------

    trainer.windowing_method = (
        WINDOWING_METHOD
    )

    trainer.calc_window = (
        CALC_WINDOW
    )

    trainer.voi_func = (
        VOI_FUNC
    )

    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    print(
        "[TEST] Cargando best.pt..."
    )

    trainer.setup_model()

    trainer.set_model_attributes()

    print(
        "[TEST] Modelo cargado"
    )

    # --------------------------------------------------------
    # CREAR DATALOADER DE TEST
    # --------------------------------------------------------

    print()
    print(
        "[TEST] Creando DataLoader de TEST..."
    )

    test_loader = trainer.get_dataloader(
        TEST_IMAGES,
        BATCH,
        rank=-1,
        mode="val",
    )

    print(
        f"[TEST] DataLoader creado: "
        f"{len(test_loader.dataset)} imágenes"
    )

    # --------------------------------------------------------
    # VALIDATOR
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "INICIANDO EVALUACIÓN SOBRE TEST V2"
    )
    print("=" * 60)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validator = DICOMValidator(
        dataloader=test_loader,
        save_dir=SAVE_DIR,
        args=trainer.args,
    )

    # --------------------------------------------------------
    # ASEGURAR PREPROCESSING V2 EN VALIDATOR
    # --------------------------------------------------------

    validator.windowing_method = (
        WINDOWING_METHOD
    )

    validator.calc_window = (
        CALC_WINDOW
    )

    validator.voi_func = (
        VOI_FUNC
    )

    # --------------------------------------------------------
    # EVALUACIÓN
    # --------------------------------------------------------

    metrics = validator(
        model=trainer.model
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TEST V2 FINALIZADO")
    print("=" * 60)

    print(metrics)

    print()
    print(
        "Preprocessing V2:"
    )

    print(
        f"  Method:       {WINDOWING_METHOD}"
    )

    print(
        f"  Calc window:  {CALC_WINDOW}"
    )

    print(
        f"  VOI function: {VOI_FUNC}"
    )

    print()
    print(
        "Resultados guardados en:"
    )

    print(
        SAVE_DIR
    )

    print("=" * 60)


if __name__ == "__main__":
    main()