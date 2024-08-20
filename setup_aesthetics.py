import seaborn as sns


#####

if __name__ == '__main__': 
    
    

    ## global settings for fontsizes in matplotlib plots
    SMALL_SIZE = 10
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 14


    rcParams = dict()
    rcParams['font.size'] = MEDIUM_SIZE          # controls default text sizes
    rcParams['axes.titlesize'] =MEDIUM_SIZE    # fontsize of the axes title
    rcParams['axes.labelsize'] =BIGGER_SIZE    # fontsize of the x and y labels
    rcParams['xtick.labelsize'] = MEDIUM_SIZE    # fontsize of the tick labels
    rcParams['ytick.labelsize'] = MEDIUM_SIZE    # fontsize of the tick labels
    rcParams['legend.fontsize'] = MEDIUM_SIZE    # legend fontsize
    rcParams['figure.titlesize'] =BIGGER_SIZE  # fontsize of the figure title

    sns.set_theme(style = 'ticks',font_scale = 1.5,rc=rcParams)

    ## set up figure grid

    FIGSIZE_A4 = (8.27, 11.69) # a4 format in inches

    FIGWIDTH_TRIPLET = FIGSIZE_A4[0]*0.3*2
    FIGHEIGHT_TRIPLET = FIGWIDTH_TRIPLET*0.75

    DPI = 300
    PAD_INCHES = 0.4

    