from rest_framework.response import Response

from rest_framework import viewsets
from .serializers import TextSerializer
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kobertmodel.settings")

import django
django.setup()
from .models import Text
from django.views import View
from django.http import HttpResponse, JsonResponse
import json
from django.core.serializers import serialize

from .models import print_list

# Create your views here.

# def index(request):
#     if request.POST['text']:
#         list=request.POST['text']
#     else:
#         list=print_list(str("기분이 좋다. 슬프다. 화난다."))
#     return HttpResponse(list)s

class IndexView(View):
    Text.objects.all().delete()
    def get(self, request):
        text = Text.objects.all()
        #data = json.loads(serialize('json', text))
        #return JsonResponse({'items': data})
        data = json.loads(serialize('json', text))
        #serializer=TextSerializer(data)
        return JsonResponse({'items': data})


    def post(self, request):
        if request.META['CONTENT_TYPE'] == "application/json":
            request = json.loads(request.body)
            serializer = TextSerializer(request['text'])
        else:
            serializer = TextSerializer(request.POST['text'])
        serializer.save()
        #return JsonResponse({'items': serializer.data})
        return Response(serializer.data, status=status.HTTP_201_CREATED)
        #return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        #https://ohgym.tistory.com/81


class TextViewset(viewsets.ModelViewSet):
    Text.objects.all().delete()

    queryset = Text.objects.all()
    serializer_class = TextSerializer


