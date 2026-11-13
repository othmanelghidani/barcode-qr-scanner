import os
import cv2
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def detect_qr(image_path, result_path):
    image = cv2.imread(image_path)

    if image is None:
        return "Erreur : image non trouvée", None

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)

    output = image.copy()

    if points is not None and data != "":
        points = points.astype(int)

        for i in range(len(points[0])):
            pt1 = tuple(points[0][i])
            pt2 = tuple(points[0][(i + 1) % len(points[0])])
            cv2.line(output, pt1, pt2, (0, 255, 0), 3)

        x = int(points[0][:, 0].min())
        y = int(points[0][:, 1].min())

        cv2.putText(
            output,
            "QR Detected",
            (x, y - 10 if y > 20 else y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imwrite(result_path, output)
        return f"QR détecté : {data}", data

    cv2.imwrite(result_path, output)
    return "Aucun QR détecté", None


def detect_barcode(image_path, result_path):
    image = cv2.imread(image_path)

    if image is None:
        return "Erreur : image non trouvée", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gradient horizontal pour faire ressortir les barres verticales
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    # Binarisation
    _, thresh = cv2.threshold(grad_x, 50, 255, cv2.THRESH_BINARY)

    # Fermeture morphologique pour relier les barres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Nettoyage
    closed = cv2.erode(closed, None, iterations=1)
    closed = cv2.dilate(closed, None, iterations=2)

    # Recherche des contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    best = None
    best_score = -1
    image_area = image.shape[0] * image.shape[1]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = w / float(h) if h != 0 else 0

        # Filtres géométriques
        if area < 0.03 * image_area:
            continue
        if ratio < 1.2:
            continue

        score = area * ratio
        if score > best_score:
            best_score = score
            best = (x, y, w, h, ratio)

    if best is not None:
        x, y, w, h, ratio = best

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            output,
            "Barcode Region",
            (x, y - 10 if y > 20 else y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        decoded_text = None

        try:
            detector = cv2.barcode_BarcodeDetector()

            # 1) Essai de décodage sur l'image entière
            result = detector.detectAndDecode(image)
            decoded_info = None

            if isinstance(result, tuple):
                if len(result) == 3:
                    decoded_info, _, _ = result
                elif len(result) == 4:
                    _, decoded_info, _, _ = result

            if decoded_info is not None and decoded_info != "":
                if isinstance(decoded_info, (list, tuple)):
                    if len(decoded_info) > 0 and str(decoded_info[0]).strip() != "":
                        decoded_text = str(decoded_info[0])
                else:
                    if str(decoded_info).strip() != "":
                        decoded_text = str(decoded_info)

            # 2) Si échec, essai sur la ROI détectée
            if not decoded_text:
                roi = image[y:y + h, x:x + w]
                result = detector.detectAndDecode(roi)
                decoded_info = None

                if isinstance(result, tuple):
                    if len(result) == 3:
                        decoded_info, _, _ = result
                    elif len(result) == 4:
                        _, decoded_info, _, _ = result

                if decoded_info is not None and decoded_info != "":
                    if isinstance(decoded_info, (list, tuple)):
                        if len(decoded_info) > 0 and str(decoded_info[0]).strip() != "":
                            decoded_text = str(decoded_info[0])
                    else:
                        if str(decoded_info).strip() != "":
                            decoded_text = str(decoded_info)

        except Exception:
            decoded_text = None

        if decoded_text:
            text_y = y + h + 30
            if text_y >= output.shape[0]:
                text_y = max(30, y - 15)

            cv2.putText(
                output,
                f"Decoded: {decoded_text}",
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            cv2.imwrite(result_path, output)
            return (
                f"Barcode localisé : x={x}, y={y}, w={w}, h={h}, ratio={ratio:.2f}\n"
                f"Barcode décodé : {decoded_text}"
            ), decoded_text

        cv2.imwrite(result_path, output)
        return (
            f"Barcode localisé : x={x}, y={y}, w={w}, h={h}, ratio={ratio:.2f}\n"
            f"Décodage impossible"
        ), None

    cv2.imwrite(result_path, output)
    return "Aucun barcode détecté", None


@app.route("/", methods=["GET", "POST"])
def index():
    result_text = None
    result_image = None
    qr_link = None
    selected_mode = "qr"
    uploaded_image = None

    if request.method == "POST":
        selected_mode = request.form.get("mode", "qr")
        file = request.files.get("image")

        if file and file.filename:
            upload_filename = file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
            file.save(upload_path)

            result_filename = "result_" + upload_filename
            result_path = os.path.join(RESULT_FOLDER, result_filename)

            uploaded_image = upload_filename

            if selected_mode == "qr":
                result_text, qr_link = detect_qr(upload_path, result_path)
            else:
                result_text, _ = detect_barcode(upload_path, result_path)

            result_image = result_filename
        else:
            result_text = "Veuillez choisir une image."

    return render_template(
        "index.html",
        result_text=result_text,
        result_image=result_image,
        qr_link=qr_link,
        selected_mode=selected_mode,
        uploaded_image=uploaded_image
    )


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)


@app.route("/uploads/<filename>")
def upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)