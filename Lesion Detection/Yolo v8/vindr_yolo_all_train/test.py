from pathlib import Path

from train_all_vindr import (
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
    "/home/enric_sena/Desktop/Mammo/runs/detect/train-11/weights/best.pt"
)

TEST_IMAGES = DATASET / "images" / "test"

DEVICE = 0

SAVE_DIR = Path(
    "/home/enric_sena/Desktop/Mammo/runs/detect/test_final"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLOv8 - TEST FINAL VinDr-Mammo")
    print("=" * 60)

    print(f"Modelo:  {MODEL_PATH}")
    print(f"Test:    {TEST_IMAGES}")
    print(f"Device:  cuda:{DEVICE}")
    print(f"Workers: {WORKERS}")
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

    n_images = len(list(TEST_IMAGES.glob("*.jpg")))

    print(f"[TEST] Imágenes encontradas: {n_images}")

    if n_images != 4000:
        raise RuntimeError(
            f"Esperaba 4000 imágenes de test, "
            f"pero hay {n_images}"
        )

    # --------------------------------------------------------
    # CREAR TRAINER
    # --------------------------------------------------------

    print()
    print("[TEST] Creando DICOMTrainer...")

    overrides = {
        "model": str(MODEL_PATH),
        "data": str(DATASET / "data.yaml"),

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

        "project": str(SAVE_DIR.parent),
        "name": SAVE_DIR.name,
        "exist_ok": True,

        "conf": 0.001,
        "iou": 0.7,
    }

    trainer = DICOMTrainer(
        overrides=overrides
    )

    # --------------------------------------------------------
    # CARGAR MODELO
    # --------------------------------------------------------

    print("[TEST] Cargando best.pt...")

    trainer.setup_model()

    trainer.set_model_attributes()

    print("[TEST] Modelo cargado")

    # --------------------------------------------------------
    # CREAR DATALOADER DE TEST
    # --------------------------------------------------------

    print()
    print("[TEST] Creando DataLoader de TEST...")

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
    print("INICIANDO EVALUACIÓN SOBRE TEST")
    print("=" * 60)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # IMPORTANTE:
    # NO usamos trainer.get_validator()
    #
    # Nuestro DICOMTrainer personalizado intenta acceder
    # a self.test_loader, que no existe.
    #
    # Creamos directamente el DICOMValidator.

    validator = DICOMValidator(
        dataloader=test_loader,
        save_dir=SAVE_DIR,
        args=trainer.args,
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
    print("TEST FINALIZADO")
    print("=" * 60)

    print(metrics)

    print()
    print(f"Resultados guardados en:")
    print(SAVE_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()
