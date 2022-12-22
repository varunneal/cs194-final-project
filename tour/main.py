import matplotlib.pyplot as plt

from image_tools import load_rgb, normalize
from TIP import TIP


def sjerome():
    sjerome = normalize(load_rgb("lib/sjerome.jpg"))
    j = TIP(sjerome)
    j.select_coords()
    model = j.get_model()
    plt.imshow(model.image)
    model.plot_rectangles()

    plt.savefig("out/sjerome5rect.jpg")

    plt.show()
    model.render()


def oxford():
    Oxford = normalize(load_rgb("lib/Oxford.jpg"))
    o = TIP(Oxford)
    o.select_coords()
    model = o.get_model(depth=500, scale=600)
    plt.imshow(model.image)
    plt.savefig("out/oxford5rect.jpg")

    plt.show()
    model.render()


def doe():
    doe_img = normalize(load_rgb("lib/doe.jpg"))
    d = TIP(doe_img)
    d.select_coords()
    model = d.get_model(scale=1000)
    model.plot_rectangles()
    plt.imshow(model.image)
    plt.savefig("out/doe5rect.jpg")

    plt.show()
    model.render()


def bay():
    bay_img = normalize(load_rgb("lib/bay.jpg"))
    b = TIP(bay_img)
    b.select_coords()
    model = b.get_model(scale=500, depth=500)
    model.plot_rectangles()
    plt.imshow(model.image)
    plt.savefig("out/bay5rect.jpg")

    plt.show()
    model.render()


def Shiva():
    Shiva_img = normalize(load_rgb("lib/Shiva.jpg"))
    s = TIP(Shiva_img)
    s.select_coords()
    model = s.get_model(scale=1000, depth=5000)
    model.plot_rectangles()
    plt.imshow(model.image)
    plt.savefig("out/Shiva5rect.jpg")

    plt.show()
    model.render()

def post():
    post_img = normalize(load_rgb("lib/post.jpg"))
    p = TIP(post_img)
    p.select_coords()
    model = p.get_model(scale=500, depth=2000)
    model.plot_rectangles()
    plt.imshow(model.image)
    plt.savefig("out/post5rect.jpg")
    model.render()

def post_alexgrey():
    post_alexgrey_img = normalize(load_rgb("lib/post_alexgrey.jpg"))
    p = TIP(post_alexgrey_img)
    p.select_coords()
    model = p.get_model(scale=500, depth=500)
    model.render()

def campanile():
    campanile_img = normalize(load_rgb("lib/campanile.jpg"))
    p = TIP(campanile_img)
    p.select_coords()
    model = p.get_model(scale=500, depth=3000)
    model.plot_rectangles()
    plt.imshow(model.image)
    plt.savefig("out/campanile5rect.jpg")
    model.render()


def doe_starrynight():
    doe = normalize(load_rgb("lib/doe_starrynight.jpg"))
    d = TIP(doe)
    d.select_coords()
    model = d.get_model(depth=500)
    model.render()


def main():
    # oxford()
    # sjerome()
    # doe()
    # bay()
    # Shiva()
    # post()
    # campanile()
    # post_alexgrey()
    doe_starrynight()


if __name__ == '__main__':
    main()
