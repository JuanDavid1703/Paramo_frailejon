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
        
def dice_coef(mask_1, mask_2, title: str, graph=False):
    mask=mask_1+mask_2
    if graph:
        plt.imshow(mask)
        plt.title(title)
        plt.show()
    mask=mask.flatten()
    total_mask=len([i for i in mask if i==1])
    rate_mask=len([i for i in mask if i==2])
    return rate_mask/(total_mask+rate_mask)


def df_dice_image(H, W, annotations_revisor, annotations_persona, pixel_ref=100):
    label_labeled=[]
    label_ref=[]
    dice_coef_list=[]
    num_obj_labeled=[]
    num_obj_ref=[]
    labeled_mask=[]
    ref_mask=[]
    
    for k_ref ,obj_r in enumerate(annotations_revisor):
        nombre_clase_r = obj_r["labels"]["labelName"]
        puntos_r = np.array(
            [[p["x"], p["y"]] for p in obj_r["content"]],
            dtype=np.int32)
        
        if "frailejon" in nombre_clase_r.lower() and "sinflor" not in nombre_clase_r.lower():
            for k_lab, obj_p in enumerate(annotations_persona):
                nombre_clase_p = obj_p["labels"]["labelName"]
                
                puntos_p = np.array(
                    [[p["x"], p["y"]] for p in obj_p["content"]],
                    dtype=np.int32)
                
                try:
                    del mask_1_rev, mask_1_pers
                except:
                    pass
                
                if "sinflor" not in nombre_clase_p.lower():
                    # print("no sinflorecencia")
                    label_labeled.append(nombre_clase_p)
                    label_ref.append(nombre_clase_r)
                    num_obj_labeled.append(k_lab)
                    num_obj_ref.append(k_ref)
    
                    
                    mask_1_rev = np.zeros((H, W), dtype=np.uint8)
                    cv.fillPoly(mask_1_rev, [puntos_r], color=1)
                    mask_1_pers = np.zeros((H, W), dtype=np.uint8)
                    cv.fillPoly(mask_1_pers, [puntos_p], color=1)
                    dice_coef_list.append(dice_coef(mask_1_pers, mask_1_rev, title="Pic"))
                    
                    ref_mask.append(cv.resize(mask_1_rev, (pixel_ref, int(pixel_ref*W/H)), interpolation=cv.INTER_NEAREST).flatten())
                    labeled_mask.append(cv.resize(mask_1_pers, (pixel_ref, int(pixel_ref*W/H)), interpolation=cv.INTER_NEAREST).flatten())
                    
                    
        
        elif "sinflor" in nombre_clase_r.lower():
            for k_lab, obj_p in enumerate(annotations_persona):
                
                nombre_clase_p = obj_p["labels"]["labelName"]
                puntos_p = np.array(
                    [[p["x"], p["y"]] for p in obj_p["content"]],
                    dtype=np.int32
                )
                try:
                    del mask_2_rev, mask_2_pers
                except:
                    pass
                
    
                if "sinflor" in nombre_clase_p.lower():
                    # print("Con sinflorecencia")
                    label_labeled.append(nombre_clase_p)
                    label_ref.append(nombre_clase_r)
                    num_obj_labeled.append(k_lab)
                    num_obj_ref.append(k_ref)
                    
                    mask_2_rev = np.zeros((H, W), dtype=np.uint8)
                    cv.fillPoly(mask_2_rev, [puntos_r], color=1)
                    mask_2_pers = np.zeros((H, W), dtype=np.uint8)
                    cv.fillPoly(mask_2_pers, [puntos_p], color=1)
                    dice_coef_list.append(dice_coef(mask_2_pers, mask_2_rev, title="Cruce en etiqueta Sinflorecencia"))
                    
                    ref_mask.append(cv.resize(mask_2_rev, (pixel_ref, int(pixel_ref*W/H)), interpolation=cv.INTER_AREA).flatten())
                    labeled_mask.append(cv.resize(mask_2_pers, (pixel_ref, int(pixel_ref*W/H)), interpolation=cv.INTER_AREA).flatten())
                    
    return label_labeled, label_ref, num_obj_labeled, num_obj_ref, dice_coef_list, labeled_mask, ref_mask
