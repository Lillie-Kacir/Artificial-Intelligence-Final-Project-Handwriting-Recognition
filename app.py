from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from handwriting_pipeline import DEFAULT_CNN_CHECKPOINT, run_pipeline


class HandwritingDesktopApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Handwriting Recognition Desktop App")
        self.root.geometry("900x650")
        self.root.minsize(820, 560)

        self.image_path_var = StringVar()
        self.output_dir_var = StringVar(value="outputs")
        self.cnn_checkpoint_var = StringVar(value=str(DEFAULT_CNN_CHECKPOINT))
        self.use_cnn_var = BooleanVar(value=True)
        self.status_var = StringVar(value="Choose an image, then click Process Image.")

        self.process_button: ttk.Button | None = None
        self.output_text: ScrolledText | None = None
        self.is_processing = False
        self.last_output_dir: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)

        title = ttk.Label(
            frame,
            text="Handwriting to Document Converter",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(anchor="w", pady=(0, 12))

        self._build_file_picker_row(
            frame,
            label_text="Handwritten image:",
            variable=self.image_path_var,
            browse_command=self._browse_image,
        )
        self._build_file_picker_row(
            frame,
            label_text="Output directory:",
            variable=self.output_dir_var,
            browse_command=self._browse_output_dir,
            browse_label="Select Folder",
        )
        self._build_file_picker_row(
            frame,
            label_text="CNN checkpoint (.pt):",
            variable=self.cnn_checkpoint_var,
            browse_command=self._browse_checkpoint,
        )

        options_frame = ttk.Frame(frame)
        options_frame.pack(fill="x", pady=(8, 12))

        ttk.Checkbutton(
            options_frame,
            text="Use CharacterCNN refinement",
            variable=self.use_cnn_var,
        ).pack(side="left")

        self.process_button = ttk.Button(
            options_frame,
            text="Process Image",
            command=self._start_processing,
        )
        self.process_button.pack(side="right")

        ttk.Label(
            frame,
            textvariable=self.status_var,
            wraplength=860,
            foreground="#1f4f92",
        ).pack(anchor="w", pady=(0, 12))

        self.output_text = ScrolledText(
            frame,
            height=22,
            width=110,
            font=("Consolas", 10),
            wrap="word",
        )
        self.output_text.pack(fill="both", expand=True)

        actions_frame = ttk.Frame(frame)
        actions_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(
            actions_frame,
            text="Open Output Folder",
            command=self._open_output_folder,
        ).pack(side="left")

        ttk.Button(
            actions_frame,
            text="Clear Output",
            command=self._clear_output,
        ).pack(side="right")

    def _build_file_picker_row(
        self,
        parent: ttk.Frame,
        label_text: str,
        variable: StringVar,
        browse_command,
        browse_label: str = "Browse",
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)

        ttk.Label(row, text=label_text, width=24).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text=browse_label, command=browse_command).pack(side="right")

    def _browse_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Handwritten Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.image_path_var.set(selected)

    def _browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Select Output Directory")
        if selected:
            self.output_dir_var.set(selected)

    def _browse_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select CharacterCNN Checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.cnn_checkpoint_var.set(selected)

    def _start_processing(self) -> None:
        if self.is_processing:
            return

        image_path = Path(self.image_path_var.get().strip())
        output_dir = Path(self.output_dir_var.get().strip() or "outputs")
        checkpoint_text = self.cnn_checkpoint_var.get().strip()

        if not image_path.exists():
            messagebox.showerror("Missing Image", "Please choose a valid handwritten image file.")
            return

        checkpoint_path = None
        if self.use_cnn_var.get():
            checkpoint_path = Path(checkpoint_text)
            if not checkpoint_path.exists():
                messagebox.showerror(
                    "Missing Checkpoint",
                    "CNN refinement is enabled but checkpoint file does not exist.",
                )
                return

        if self.process_button is not None:
            self.process_button.configure(state="disabled")
        self.is_processing = True
        self.status_var.set("Processing image... this can take a few seconds.")

        thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(image_path, output_dir, checkpoint_path),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_worker(
        self,
        image_path: Path,
        output_dir: Path,
        checkpoint_path: Path | None,
    ) -> None:
        try:
            output_stem = output_dir / "predicted_words"
            raw_text, predicted_text, txt_path, docx_path = run_pipeline(
                image_path=image_path,
                output_stem=output_stem,
                cnn_checkpoint=checkpoint_path,
            )
            self.root.after(
                0,
                lambda: self._on_pipeline_success(raw_text, predicted_text, txt_path, docx_path),
            )
        except Exception as error:  # noqa: BLE001
            self.root.after(0, lambda: self._on_pipeline_error(str(error)))

    def _on_pipeline_success(
        self,
        raw_text: str,
        predicted_text: str,
        txt_path: Path,
        docx_path: Path,
    ) -> None:
        self._set_idle_state()
        self.last_output_dir = txt_path.parent
        self.status_var.set(
            f"Done. Saved text to {txt_path} and document to {docx_path}."
        )

        if self.output_text is None:
            return
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", "Raw OCR text:\n")
        self.output_text.insert("end", raw_text.strip() + "\n\n")
        self.output_text.insert("end", "Predicted text:\n")
        self.output_text.insert("end", predicted_text.strip() + "\n")

    def _on_pipeline_error(self, message: str) -> None:
        self._set_idle_state()
        self.status_var.set("Processing failed. Check error details.")
        messagebox.showerror("Processing Error", message)

    def _set_idle_state(self) -> None:
        self.is_processing = False
        if self.process_button is not None:
            self.process_button.configure(state="normal")

    def _open_output_folder(self) -> None:
        folder = self.last_output_dir or Path(self.output_dir_var.get().strip() or "outputs")
        folder.mkdir(parents=True, exist_ok=True)
        self._open_in_file_explorer(folder)

    def _clear_output(self) -> None:
        if self.output_text is not None:
            self.output_text.delete("1.0", "end")
        self.status_var.set("Output cleared. Ready for another image.")

    @staticmethod
    def _open_in_file_explorer(path: Path) -> None:
        absolute_path = str(path.resolve())
        if sys.platform.startswith("win"):
            os.startfile(absolute_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", absolute_path], check=False)
        else:
            subprocess.run(["xdg-open", absolute_path], check=False)


def main() -> None:
    root = Tk()
    ttk.Style(root).theme_use("clam")
    HandwritingDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
