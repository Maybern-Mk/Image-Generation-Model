# import streamlit as st 
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import io 

# st.set_page_config(page_title="GAN Generator",layout="wide")

# st.markdown("""
#     <style>
#     .main { background-color: #0E1117; }   /* Background color */
#     .title { font-size: 40px; font-weight: bold; color: #00FFD1; }  /* Title style */
#     .subtitle { font-size: 18px; color: #AAAAAA; }  /* Subtitle style */
#     .stButton>button {
#         background-color: #00FFD1;  /* Button color */
#         color: black;
#         border-radius: 10px;        /* Rounded button */
#         height: 50px;
#         width: 100%;
#         font-size: 16px;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="title">Gan Image Generator</div>',unsafe_allow_html=True)

# st.markdown('<div class="subtitle">Generate Handwritten digits using AI</div>',unsafe_allow_html=True)

# latent_dim=100

# st.sidebar.title("👾Contols")

# num_images=st.sidebar.slider("Number of Images",1,5,1)

# noise_scale=st.sidebar.slider("Noise Variation",0.5,2.0,1.0)

# st.sidebar.markdown("---")
# st.sidebar.info("Change Settings and click generate!")

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.model=nn.Sequential([
#             nn.Linear=nn.Linear(latent_dim,256),# Noise
#             nn.ReLU(),                          #Activation
#             nn.Linear(256,512),                 #Increase features
#             nn.ReLU(),                          
#             nn.Linear(512,784),                 #Output
#             nn.Tanh()                           #Output between -1 tO 1
         
#      ])
#         def forward(self,z):
#             return self.model(z)

# @st.cache_resource
# def load_model():
#     model=Generator()
    
#     model.load_state_dict(
#         torch.load("generator.pth",map_location=torch.device("cpu"))

#     )
# model.eval()
# return load_model

# model=load_model()

# if st.button("Generate Images"):
#     cols=st.columns(num_images)
#     for i in range(num_images):
#         z=torch.randn(1,latent_dim)*noise_scale
#         gen_image=model(z).detach().numpy().reshape(28,28)
#         fig.ax=plt.subplots()
#         ax.imshow(gen_img,cmap="gray")
#         ax.axis("off")
#         cols[i].pyplot(fig)
        
#         img=(gen_img*127.5+127.5).astype(np.uint8)
        
#         pil_img=Image.fromarray(img)
        
#         buf=io.BytesIO()
        
#         pil_img.save(buf,format="PNG")
        
#         cols[i].download_button(
#             label="1 Download",
#             data=buf.getvalue(),
#             file_name=f"gan_digit_{i}.png",
#             mime="image?png"
            
#         )
        
# st.divider()
# st.caption("Built with ❤️ using GAN+Pytorch+Streamlit")

import streamlit as st 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io 

st.set_page_config(page_title="GAN Generator", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .title { font-size: 40px; font-weight: bold; color: #00FFD1; }
    .subtitle { font-size: 18px; color: #AAAAAA; }
    .stButton>button {
        background-color: #00FFD1;
        color: black;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">GAN Image Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate Handwritten digits using AI</div>', unsafe_allow_html=True)

latent_dim = 100

st.sidebar.title("👾 Controls")
num_images = st.sidebar.slider("Number of Images", 1, 5, 1)
noise_scale = st.sidebar.slider("Noise Variation", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.info("Change settings and click generate!")

# ✅ FIXED GENERATOR
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# ✅ FIXED MODEL LOADING
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(
        torch.load("generator.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

model = load_model()

# ✅ GENERATE BUTTON
if st.button("Generate Images"):
    cols = st.columns(num_images)

    for i in range(num_images):
        z = torch.randn(1, latent_dim) * noise_scale
        gen_img = model(z).detach().numpy().reshape(28, 28)

        fig, ax = plt.subplots()
        ax.imshow(gen_img, cmap="gray")
        ax.axis("off")

        cols[i].pyplot(fig)

        # Convert to image
        img = (gen_img * 127.5 + 127.5).astype(np.uint8)
        pil_img = Image.fromarray(img)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")

        cols[i].download_button(
            label="Download",
            data=buf.getvalue(),
            file_name=f"gan_digit_{i}.png",
            mime="image/png"
        )

st.divider()
st.caption("Built with ❤️ using GAN + PyTorch + Streamlit")
