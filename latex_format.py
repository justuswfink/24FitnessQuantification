## this code is from a StackOverflow answer
## https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python

def float2latex(f, display = ".2e"):
    format_template = "{0:" + display + "}"
    float_str = format_template.format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base == 1:
            return r"{10^{{{1}}}".format(base, int(exponent))
        else:
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


if __name__ == '__main__': 
    print("You can call this function as follows")
    print(">>> float2latex(1.12345e-12, display = '.2e')")
    print(float2latex(1.12345e-12))


    