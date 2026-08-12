from matplotlib.pyplot import imread, imsave
from numpy import median
import os
from pandas import read_excel

def crop_image(image_path, crop_width, crop_height, trheshold=1, save_dir=None):
    """
    Divide una imagen en recortes de tamaño fijo.

    Parameters
    ----------
    image_path : str
        Ruta de la imagen.
    crop_width : int
        Ancho de cada recorte.
    crop_height : int
        Alto de cada recorte.
    save_dir : str, optional
        Carpeta donde guardar los recortes.

    Returns
    -------
    list
        Lista de imágenes recortadas (numpy arrays).
    """

    image = imread(image_path)[:,:,:3]

    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    h, w = image.shape[:2]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    crops = []
    crop_id = 0

    for y in range(0, h, crop_height):
        for x in range(0, w, crop_width):

            # Evitar recortes incompletos
            if y + crop_height > h or x + crop_width > w:
                continue

            crop = image[y:y + crop_height, x:x + crop_width]

            if median(crop)<trheshold:
                crops.append(crop)

                if save_dir:
                    filename = os.path.join(save_dir, f"crop_{crop_id}.jpg")
                    imsave(filename, crop)

            crop_id += 1

    return crops

def cut_list(path: str, path_reference: str):
    list_file=os.listdir(path)
    df_measure=read_excel(path_reference)
    for f in list_file:
        name_file=f.split(sep=".")[0]
        try:
            m=df_measure[df_measure["Nombre parcela"]==name_file].values[0]
        except:
            m=int(6500/df_measure["mm/pix"].mean())
        crops= crop_image(path+f, crop_height=m, crop_width= m, trheshold=0.65, save_dir="recortes/"+f)