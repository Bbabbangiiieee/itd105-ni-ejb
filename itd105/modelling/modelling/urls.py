"""
URL configuration for modelling project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ml import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login, name='login'),
    path('/register/', views.register, name='register'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    path('/signout/', views.signout, name='signout'),

    path('/eda/', views.eda, name='eda'),
    path('/class_add/', views.class_add_view, name='class_add'),
    path('add_class/', views.add_class, name='add_class'),
    path('/train_class/', views.train_class, name='train_class'),
    path('model_train_class/', views.model_train_class, name='model_train_class'),
    path('predict_class/', views.predict_class, name='predict_class'),

    path('/eda_reg/', views.eda_reg, name='eda_reg'),
    path('/reg_add/', views.reg_add_view, name='reg_add'),
    path('add_reg/', views.add_reg, name='add_reg'),
    path('/train_reg/', views.train_reg, name='train_reg'),
    path('model_train_reg/', views.model_train_reg, name='model_train_reg'),
    path('predict_reg/', views.predict_reg, name='predict_reg'),
]
