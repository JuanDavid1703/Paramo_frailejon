from matplotlib.pyplot import imread, imsave
from numpy import median
import os
from pandas import read_excel
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import json

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


def df_dice_image(H, W, annotations_revisor, annotations_persona, pixel_ref=100, interpolation=cv.INTER_LANCZOS4):
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
        
        if "frail" in nombre_clase_r.lower() and "sinflor" not in nombre_clase_r.lower():
            for k_lab, obj_p in enumerate(annotations_persona):
                nombre_clase_p = obj_p["labels"]["labelName"]
                
                puntos_p = np.array(
                    [[p["x"], p["y"]] for p in obj_p["content"]],
                    dtype=np.int32)
                
                try:
                    del mask_1_rev, mask_1_pers, ref_mask_resized, labeled_mask_resised
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
                    
                    ref_mask_resized=cv.resize(mask_1_rev, (pixel_ref, int(pixel_ref*W/H)), interpolation=interpolation)
                    ref_mask_resized=ref_mask_resized.reshape(-1)
                    ref_mask.append(ref_mask_resized)
                    labeled_mask_resised=cv.resize(mask_1_pers, (pixel_ref, int(pixel_ref*W/H)), interpolation=interpolation)
                    labeled_mask_resised=labeled_mask_resised.reshape(-1)
                    labeled_mask.append(labeled_mask_resised)
                    
                    
        
        elif "sinflor" in nombre_clase_r.lower():
            for k_lab, obj_p in enumerate(annotations_persona):
                
                nombre_clase_p = obj_p["labels"]["labelName"]
                puntos_p = np.array(
                    [[p["x"], p["y"]] for p in obj_p["content"]],
                    dtype=np.int32
                )
                try:
                    del mask_2_rev, mask_2_pers, ref_mask_resized, labeled_mask_resised
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
                    
                    ref_mask_resized=cv.resize(mask_2_rev, (pixel_ref, int(pixel_ref*W/H)), interpolation=interpolation)
                    ref_mask_resized=ref_mask_resized.reshape(-1)
                    ref_mask.append(ref_mask_resized)
                    labeled_mask_resised=cv.resize(mask_2_pers, (pixel_ref, int(pixel_ref*W/H)), interpolation=interpolation)
                    labeled_mask_resised=labeled_mask_resised.reshape(-1)
                    labeled_mask.append(labeled_mask_resised)
                    
                    
    return label_labeled, label_ref, num_obj_labeled, num_obj_ref, dice_coef_list, labeled_mask, ref_mask

def ver_mascara(path_data_base:str, name_pic:str, pixel_rate=256):

    df=pd.read_csv(path_data_base, sep=";")
    df["label_image_reduced"]=df["label_image_reduced"].apply(lambda x: x.split(","))
    df["ref_image_reduced"]=df["ref_image_reduced"].apply(lambda x: x.split(","))
    df["label_image_reduced"]=df["label_image_reduced"].apply(lambda x: np.uint8(x))
    df["ref_image_reduced"]=df["ref_image_reduced"].apply(lambda x: np.uint8(x))

    df_filtered=df[df["pic_name"]==name_pic]

    for num in df_filtered["Num_obj_ref"]:
        label_ima=df_filtered[df_filtered["Num_obj_ref"]==num]["label_image_reduced"].values[0]
        label_ima=label_ima.reshape(len(label_ima)//pixel_rate, pixel_rate)
        ref_ima=df_filtered[df_filtered["Num_obj_ref"]==num]["ref_image_reduced"].values[0]
        ref_ima=ref_ima.reshape(len(ref_ima)//pixel_rate, pixel_rate)
        coef_dice=df_filtered[df_filtered["Num_obj_ref"]==num]["Dice_coef"].values[0].round(2)
        label=df_filtered[df_filtered["Num_obj_ref"]==num]["Label_labeled"].values[0]

        fig, ax= plt.subplots(1, 2, figsize=(10, 5)) 
        ax[0].imshow(label_ima)
        ax[0].set_title("Labeled image")
        ax[1].imshow(ref_ima)
        ax[1].set_title("Reference image")
        # Título general
        fig.suptitle(
            f"coeficiente de dice {str(coef_dice)} de la imagen {name_pic} y el objeto numero {num} con etiqueta {label}",
            fontsize=14,
            fontweight="bold"
        )

        # Ajustar espacio para que no se superpongan los títulos
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def data_analysis(path_person:str, save:bool=True):
    list_person=os.listdir(path_person)
    list_person=[f for f in list_person if not f.endswith(".csv")]
    list_images=os.listdir(os.path.join(path_person,list_person[0]))
    df_general=pd.DataFrame()

    for ima_name in list_images:
        ima=plt.imread(os.path.join(path_person,list_person[0],ima_name))
        # Altura y ancho de la imagen
        Height, Width = ima.shape[:2]
        df_pic=pd.DataFrame()
        #Nombre de la imagen sin extensión y sin prefijo
        pic_name=ima_name.split(".")[0].split("_")[-1]
        
        #Nombre de los json de la persona y del revisor que corresponden a la imagen
        json_referenced_name=[ i for i in os.listdir(os.path.join(path_person,list_person[2])) if pic_name in i ][0]
        json_labeled_name=[ i for i in os.listdir(os.path.join(path_person,list_person[1])) if pic_name in i ][0]
        
        #apertura de los json de la persona y del revisor que corresponden a la imagen
        with open(os.path.join(path_person,list_person[2],json_referenced_name)) as f:
            annotations_revisor = json.load(f)

        with open(os.path.join(path_person,list_person[1],json_labeled_name)) as f:
            annotations_persona = json.load(f)
        
        label_labeled, label_ref, num_obj_labeled, num_obj_ref, dice_coef_list, labeled_mask, ref_mask=df_dice_image(Height, Width, 
                                                                                                                    annotations_revisor,
                                                                                                                    annotations_persona,
                                                                                                                    pixel_ref=256)
        
        df_pic["Name_labeler"]=[path_person]*len(label_labeled)
        df_pic["pic_name"]=[pic_name]*len(label_labeled)
        df_pic["Num_obj_ref"]=num_obj_ref
        df_pic["Num_obj_labeled"]=num_obj_labeled
        df_pic["Label_ref"]=label_ref
        df_pic["Label_labeled"]=label_labeled
        df_pic["Dice_coef"]=dice_coef_list
        df_pic["label_image_reduced"]=labeled_mask
        df_pic["ref_image_reduced"]=ref_mask
        df_pic["Dimesion_image"]=[(Height, Width)] * len(label_labeled)
        
        df_general=pd.concat([df_general, df_pic], ignore_index=True)

    df_general["num_parcela"]=df_general["pic_name"].map(lambda x: x.split("C")[0])
    df_general.sort_values(by=["pic_name","Num_obj_ref", "Dice_coef"], ascending=[True, True, False],inplace=True)
    df_general.drop_duplicates(subset=["pic_name","Num_obj_ref", "Label_ref"], inplace=True)
    df_general.reset_index(drop=True, inplace=True)
    df_general["label_image_reduced"]=df_general["label_image_reduced"].apply(lambda x: ",".join([str(i) for i  in x.tolist()]))
    df_general["ref_image_reduced"]=df_general["ref_image_reduced"].apply(lambda x: ",".join([str(i) for i  in x.tolist()]) )

    if save:
        df_general.to_csv(path_person+ f"/df_{path_person}.csv", index=False, sep=";")
    return df_general
