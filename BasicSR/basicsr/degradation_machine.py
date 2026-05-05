import cv2
import numpy as np
import random
import os
from scipy.ndimage import gaussian_filter

class DegradationMachine:
    def __init__(self):
        # Rasponi za stohastičko modeliranje
        self.blur_range = (0.1, 1.1)
        self.noise_range = (1, 15)
        self.jpeg_range = (30, 95)
        self.sinc_range = (7, 15)

    def _generate_random_mask(self, shape):
        """Generira meku masku za prostorno varijabilne efekte."""
        mask = np.zeros(shape[:2], dtype=np.float32)
        for _ in range(random.randint(2, 5)):
            cx, cy = random.randint(0, shape[1]), random.randint(0, shape[0])
            sigma = random.randint(50, 150)
            y, x = np.ogrid[-cy:shape[0]-cy, -cx:shape[1]-cx]
            mask += np.exp(-(x*x + y*y) / (2. * sigma**2))
        return np.clip(mask, 0, 1)[..., None]

    def _generate_sinc_kernel(self, size, cutoff):
        """Generira Sinc kernel za simulaciju 'ringing' artefakata."""
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        r = np.sqrt(xx**2 + yy**2) + 1e-8
        kernel = np.sin(np.pi * cutoff * r) / (np.pi * r)
        return kernel / np.sum(kernel)

    # --- KEMIJSKI I OPTIČKI MODULI ---
    def apply_vignetting(self, img):
        """Dodaje optičko zatamnjenje rubova leće."""
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/1.5)
        kernel_y = cv2.getGaussianKernel(rows, rows/1.5)
        mask = (kernel_y * kernel_x.T)
        mask = 1 - random.uniform(0.3, 0.7) * (1 - mask / mask.max())
        return (img * mask[..., None]).astype(np.uint8)

    def apply_color_aging(self, img):
        """Simulira kemijsko blijeđenje - SUPTILNO da model ne uči micati boje."""
        # Blagi gamma - manji raspon da ne mijenja boje previše
        gamma = random.uniform(0.9, 1.15)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        
        # Sepia: RIJETKO (30%) i s VARIJABILNIM intenzitetom (20-60%)
        # Tako model ne nauči "sepia = loše, ukloni"
        if random.random() > 0.7:  # Samo 30% šansa (bilo 60%)
            sepia_matrix = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
            sepia_strength = random.uniform(0.2, 0.6)  # Varijabilni intenzitet
            sepia_img = cv2.transform(img, sepia_matrix)
            img = (img * (1 - sepia_strength) + sepia_img * sepia_strength)
        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_chemical_foxing(self, img):
        """Simulira smeđe mrlje starosti (foxing)."""
        out = img.copy()
        h, w = img.shape[:2]
        for _ in range(random.randint(3, 7)):
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = random.randint(0, w), random.randint(0, h)
            axes = (random.randint(10, 25), random.randint(10, 25))
            cv2.ellipse(mask, (cx, cy), axes, random.randint(0, 360), 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            foxing_color = np.array([30, 60, 110]) # BGR smeđa
            alpha = (mask / 255.0 * random.uniform(0.2, 0.5))[..., None]
            out = (out * (1 - alpha) + foxing_color * alpha).astype(np.uint8)
        return out

    # --- FIZIČKA OŠTEĆENJA ---
    def add_physical_artifacts(self, img):
        """Dodaje ogrebotine, prašinu i mrlje."""
        out = img.copy()
        for _ in range(random.randint(1, 6)):
            p1 = (random.randint(0, out.shape[1]), random.randint(0, out.shape[0]))
            p2 = (random.randint(0, out.shape[1]), random.randint(0, out.shape[0]))
            cv2.line(out, p1, p2, random.choice([(200, 200, 200), (40, 40, 40)]), random.randint(1, 2))
        for _ in range(random.randint(20, 60)):
            cv2.circle(out, (random.randint(0, out.shape[1]), random.randint(0, out.shape[0])), 
                       random.randint(1, 2), (10, 10, 10), -1)
        return out

    def apply_spatial_degradation(self, img):
        """Zamućenje i šum - SMANJENI BLUR da model ne uči zamagljivati."""
        mask = self._generate_random_mask(img.shape)
        # KRITIČNO: Sigma 0.3-1.5 umjesto 2.0-4.0
        # Previsok blur = model uči da su oštri rubovi "šum" koji treba ukloniti
        blur_sigma = random.uniform(0.3, 1.5)
        blurred = gaussian_filter(img, sigma=(blur_sigma, blur_sigma, 0))
        img = (img * (1 - mask) + blurred * mask).astype(np.uint8)
        sigma = random.uniform(*self.noise_range)
        noise = np.random.normal(0, sigma, img.shape)
        img = np.clip(img + noise * self._generate_random_mask(img.shape), 0, 255).astype(np.uint8)
        return img

    def apply_digital_noise(self, img):
        """Poissonov šum digitalizacije."""
        vals = 2 ** np.ceil(np.log2(len(np.unique(img))))
        out = np.random.poisson(img * vals) / float(vals)
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_jpeg(self, img):
        """JPEG artefakti."""
        q = random.randint(*self.jpeg_range)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, 1)
    
    
    def run(self, img_hr):
        """
        Glavni pipeline koji transformira HR sliku u LR (4x manju)
        koristeći model degradacije visokog reda.
        """
        # --- 1. KRUG: OPTIKA I KEMIJA (Originalna rezolucija) ---
        out = self.apply_vignetting(img_hr)
        out = self.apply_color_aging(out)
        out = self.apply_chemical_foxing(out)
        out = self.add_physical_artifacts(out)
        
        # Prvo zamućenje i smanjenje (1. red degradacije)
        out = self.apply_spatial_degradation(out)
        out = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        
        # --- 2. KRUG: DIGITALIZACIJA (Smanjena rezolucija) ---
        out = self.apply_digital_noise(out)
        out = self.apply_jpeg(out)
        
        # Drugo smanjenje (2. red degradacije -> ukupno 4x)
        out = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        # --- 3. FINALNI SINC FILTAR (Ringing artefakti) ---
        # Smanjeno na 40% (bilo 80%) - previše ringing = model zaglađuje
        if random.random() > 0.6:
            cutoff = random.uniform(0.15, 0.4)  # Manji cutoff = manje agresivan
            kernel_size = random.choice([7, 9, 11])
            sinc_kernel = self._generate_sinc_kernel(kernel_size, cutoff)
            out = cv2.filter2D(out, -1, sinc_kernel)
            
        return out










class UltimateDegraderWithStoryboard(DegradationMachine):
    def run_with_storyboard(self, hr_img):
        """Sprema slike nakon svake ključne faze za znanstvenu analizu."""
        storyboard = {}
        storyboard['0_Original_HR'] = hr_img.copy()

        # 1. FAZA: Optika i kemija
        out = self.apply_vignetting(hr_img)
        storyboard['1_Vignetting'] = out.copy()
        
        out = self.apply_color_aging(out)
        storyboard['2_Color_Aging'] = out.copy()

        # 2. FAZA: Fizičko propadanje
        out = self.apply_chemical_foxing(out) 
        storyboard['3_Foxing_Stains'] = out.copy()
        
        out = self.add_physical_artifacts(out)
        storyboard['4_Physical_Damage'] = out.copy()

        # 3. FAZA: Optičko zamućenje i smanjenje
        out = self.apply_spatial_degradation(out)
        storyboard['5_Spatial_Blur'] = out.copy()
        
        out = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        storyboard['6_First_Resize_2x'] = out.copy()

        # 4. FAZA: Digitalizacija
        out = self.apply_digital_noise(out)
        storyboard['7_Digital_Noise'] = out.copy()
        
        out = self.apply_jpeg(out)
        storyboard['8_JPEG_Compression'] = out.copy()

        # 5. FAZA: Drugi red i Sinc
        out = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        out = cv2.filter2D(out, -1, self._generate_sinc_kernel(15, 0.2))
        storyboard['9_Final_LR_4x'] = out.copy()

        return storyboard

def save_storyboard(hr_path, output_dir):
    if not os.path.exists(hr_path):
        print(f"Greška: Datoteka {hr_path} ne postoji!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    hr_img = cv2.imread(hr_path)
    degrader = UltimateDegraderWithStoryboard()
    
    results = degrader.run_with_storyboard(hr_img)
    
    for name, img in results.items():
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img)
    
    print(f"Storyboard generiran u folderu: {output_dir}")

if __name__ == "__main__":
    # Ovdje stavi naziv svoje slike
    save_storyboard("original_hr.jpg", "storyboard_output")