# Covid-19 models

To run:

```
virtualenv -p python3 /tmp/venv
source /tmp/venv/bin/activate
pip install -r requirements.txt

./fit.py fetch
./fit.py show-map
```

The data that it downloads is https://github.com/CSSEGISandData/COVID-19. I tried an exponential model (and the code is still in there as an option), but since the curves tend to go non-exponential when social distancing is enacted, it gets pretty tweaky and inaccurate.

Ultimately, I abandoned fancy curve fitting and just linearly interpolated the Italy data directly.

In other words, if a state has 500 cases, the code looks up what day Italy had ~500 cases, and then what day Italy had ~20,000 cases, and takes the difference. That's what's reported in the "T-whatever days" thing in the image.


*states_21basic.zip* comes from https://www.arcgis.com/home/item.html?id=f7f805eb65eb4ab787a0a3e1116ca7e5.

![](https://www.dropbox.com/s/a5gc5ybe6juilbb/2020-03-18.png?raw=1)



