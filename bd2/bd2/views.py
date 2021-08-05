import json
import os
import glob
from django.shortcuts import render
from .forms import ImageForm

def mainpage(request):
  return render(request,'index.html')

def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            imagefilename = glob.glob("./media/images/*")[0]
            message = "Subido con éxito"
            os.remove(imagefilename)
            print(imagefilename, " eliminado")
            return render(request, 'dashboard.html', {'message': message})
    else:
        form = ImageForm()
    return render(request, 'dashboard.html', {'form': form})
#def uploadimage(request):