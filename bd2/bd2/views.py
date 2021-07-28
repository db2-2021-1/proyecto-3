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
            # Get the current instance object to display in the template
            message = "Subido con éxito"
            return render(request, 'dashboard.html', {'message': message})
    else:
        form = ImageForm()
    return render(request, 'dashboard.html', {'form': form})
#def uploadimage(request):