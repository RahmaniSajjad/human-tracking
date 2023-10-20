# داکیومنت پروژه: تشخیص و تجزیه توصیف جسم انسان با استفاده از OpenCV و MediaPipe

## توضیحات کلی:
این پروژه یک اسکریپت پایتون است که از کتابخانه‌های OpenCV و MediaPipe برای تشخیص و تجزیه توصیف جسم انسان (پوز) در زمان واقعی از یک وبکم یا ویدئو استفاده می‌کند. این اسکریپت پوز لندمارک‌ها را استخراج کرده و موقعیت‌های نسبی برای اعضای خاص بدن محاسبه می‌کند. همچنین این امکان را فراهم می‌کند تا اسکریپت در حالت تست نمایش تصویر تشخیص‌یافته را نمایش دهد و اطلاعات در قالب JSON ذخیره کند.

## نصب محتوای مورد نیاز:
- نصب کتابخانه OpenCV:
```diff
pip install opencv-python
```
- نصب کتابخانه MediaPipe:
```diff
pip install mediapipe
```

## نحوه استفاده:
برای استفاده از این اسکریپت، مراحل زیر را انجام دهید:

### اجرای اسکریپت:
- برای اجرای اسکریپت، از ترمینال یا محیط اجرایی پایتون خود دستور زیر را اجرا کنید:
```diff
python main_WebCam_INPUT.py
```
### در حالت تست:
- در حالت تست (`isTestMode` تنظیم شده به `True` در کد)، اسکریپت تصویر تشخیص‌یافته را به صورت نمایشی نمایش داده و مختصات نسبی را در کنسول چاپ می‌کند. برای خروج از حالت تست، دکمه 'q' را فشار دهید.

### در حالت عملیاتی:
- در حالت عملیاتی، اسکریپت تعداد تعیین شده از فریم‌های ورودی را پردازش می‌کند و اطلاعات موقعیت نسبی را در قالب JSON ذخیره می‌کند. تعداد فریم‌های مورد نظر را در متغیر `frame_count` تعیین کنید.

## نمونه خروجی:
- در حالت تست: اطلاعات موقعیت نسبی برای اعضای مختلف بدن در هر فریم نمایش داده می‌شود.
- در حالت عملیاتی: اطلاعات موقعیت نسبی برای اعضای مختلف بدن در تمام فریم‌ها در یک فایل JSON ذخیره می‌شوند.

## نحوه تغییر تنظیمات:
شما می‌توانید تنظیمات مختلفی از جمله نمایش یا عدم نمایش تصویر، تعداد فریم‌ها، نوع اطلاعات محاسبه شده و موارد دیگر را در کد اصلی اسکریپت تغییر دهید.

## تنظیمات اختیاری:
- در کد اصلی اسکریپت، شما می‌توانید تنظیمات مختلفی را تغییر دهید:
<div dir="rtl">
  
  - `isTestMode`: مشخص می‌کند که آیا اسکریپت در حالت تست (نمایش تصویر و چاپ مختصات) باشد یا نه.
    
  - `frame_count`: تعداد فریم‌هایی که در حالت عملیاتی پردازش می‌شوند.
    
  - `نوع اطلاعات محاسبه شده`: می‌توانید تعیین کنید که کدام اطلاعات محاسبه شود و به چه صورت.
</div>

## استفاده از این پروژه:
شما می‌توانید این اسکریپت را برای موارد مختلفی استفاده کنید، از جمله:
- پیش‌بینی حرکات بدن در ورزش یا تمرینات ورزشی.
- پیاده‌سازی کنترل حرکتی بر روی اپلیکیشن‌ها یا بازی‌های ویدئویی.
- ضبط و تجزیه و تحلیل جلسات آموزشی یا کلاس‌های تمرین.

## مراحل بعدی:
این پروژه به عنوان یک شروع کارآمد می‌تواند توسعه یابد. مراحل بعدی می‌توانند عبارت باشند از:
- ذخیره اطلاعات در پایگاه داده برای تجزیه و تحلیل بیشتر.
- پیاده‌سازی یک رابط کاربری گرافیکی برای نمایش اطلاعات به کاربران.
- استفاده از این پروژه در زمینه‌های پزشکی برای تشخیص مشکلات جسمانی.

## نکات پایانی:
- این پروژه نشان می‌دهد چگونه از تکنولوژی‌های تشخیص حرکات و توصیف جسم انسان برای موارد مختلفی استفاده کنید.
- شما می‌توانید این پروژه را برای توسعه، تجزیه و تحلیل حرکات بدن، یادگیری ماشینی و بیشتر بهره ببرید.

## مراجع:
- [لینک به مستندات OpenCV](https://docs.opencv.org/).
- [لینک به مستندات MediaPipe](https://mediapipe.dev/).

## خاتمه:
این پروژه به شما امکان می‌دهد تا بهره‌برداری مبتکرانه از تکنولوژی‌های تشخیص حرکات و توصیف جسم انسان داشته باشید. ما امیدواریم که این پروژه برای شما مفید باشد و شما بتوانید از آن بهره‌وری کنید.
