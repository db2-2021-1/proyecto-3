from PIL import Image
import json
import os
import glob
from django.shortcuts import render
from .forms import ImageForm
from .query import knn_search, range_search, load_index
from face_recognition.api import load_image_file, face_encodings

index = load_index("rtree", 128)

def mainpage(request):
  return render(request,'index.html')
            
def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data = form.cleaned_data
            print(data['title'])
            #knn 
            #range

            imagefilename = glob.glob("./media/images/*")[0]
            message = "Subido con éxito"
            face = face_encodings(load_image_file(imagefilename))[0]
            result = []
            if data['title']=="knn" :            
                result = knn_search(index, face, 3)
            elif data['title']=="range":
                result = range_search(index, face, 3.0)
            imagelist = []
            result = list(dict.fromkeys(result))
            for id in result:
                images = glob.glob("./static/lfw/"+id+"/*")
                if len(images)>0:
                    print(images[0])
                    imagelist.append(images[0])
            os.remove(imagefilename)
            return render(request, 'dashboard.html',  {'form': form, 'imagelist': imagelist})
    else:
        form = ImageForm()
    return render(request, 'dashboard.html', {'form': form})
#def uploadimage(request):
