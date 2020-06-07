import  numpy as np
from django import template
register = template.Library()

@register.filter
def IsNan(value):
    if(np.isnan(value) == True):
        value = -10000
    return value