from rest_framework import  serializers
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kobertmodel.settings")

import django
django.setup()
from .models import Text
from .models import print_list

class TextSerializer(serializers.ModelSerializer):
    class Meta:
        model = Text
        fields = ('text',)

    def create(self, validated_data):
        #list=print_list(str(self.validated_data['text']))
        self.validated_data['text']=print_list(self.validated_data['text'])
        return Text.objects.create(**self.validated_data)
