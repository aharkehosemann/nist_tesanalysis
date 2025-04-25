
from bolotest_routines import *

### testing suite - test model output
def test_objs(lw=5, ll=220, dsub=0.400, dSiOx=0.120, w1w=5, w2w=3, dW1=0, dI1=0, dW2=0, dI2=0, d_stacks=np.array([0, 0, 0, 0]), w_stacks=np.array([0, 0, 0, 0, 0]),
                model='Three-Layer', stack_I=False, stack_N=False, tall_Istacks=False, constrained=False, supG=0.0, calc='Median', deltalw_CD=0.0, extend_I2=False, calc_deff=False):
    ### create dummy objects for testing

    test_bolo = {}; test_bolo['geometry'] = {}

    test_bolo = {}; test_bolo['geometry'] = {}
    test_bolo['geometry']['ll']       = ll   # [um] bolotest leg length
    test_bolo['geometry']['lw']       = lw   # [um] bolotest leg width
    test_bolo['geometry']['layer_ds'] = np.array([dsub, dsub, dsub, dsub+0.1, dsub-0.1, dsub, dW1, dW1, dI1, dI1, dI1, dW2, dW2, dI2, dI2, dI1+dI2])
    test_bolo['geometry']['dsub']     = dsub   # dS for leg A
    test_bolo['geometry']['w1w']      = w1w   # [um] W1 width
    test_bolo['geometry']['w2w']      = w2w   # [um] W2 width
    test_bolo['geometry']['dW1']      = dW1; test_bolo['geometry']['dW2'] = dW2
    test_bolo['geometry']['dI1']      = dI1; test_bolo['geometry']['dI2'] = dI2   # [um] leg A film thicknesses
    test_bolo['geometry']['La']       = ll;  test_bolo['geometry']['acoustic_Lscale'] = True   # bolotest acoustic length - shouldn't matter because acoustic scaling ratio should be 1
    test_bolo['geometry']['dSiOx']    = dSiOx   # [um] SiOx thickness of substrate layer
    test_bolo['geometry']['d_stacks'] = d_stacks   # [um] layer thickness adjustments near W1 and W2 edges
    test_bolo['geometry']['w_stacks'] = w_stacks   # [um] widths of layer thickness adjustment regions
    test_bolo['geometry']['deltalw_CD'] = deltalw_CD   # [um] legs A and D may be skinnier than B, C, E, F, G

    test_anopts = {}
    test_anopts['stack_I']        = stack_I   # account for I1-I2 stacks in areas wider than W2
    test_anopts['stack_N']        = stack_N   # account for S-I1-I2 stacks in areas wider than W1
    test_anopts['tall_Istacks']   = tall_Istacks   # account for taller I layers at W layer edges
    test_anopts['extend_I2']      = extend_I2   # [um] widths of layer thickness adjustment regions
    test_anopts['constrained']    = constrained   # use constrained model results
    test_anopts['model']          = model   # Two-, Three-, or Four-Layer model?
    test_anopts['supG']           = supG    # reduce G for substrate on legs B, E & G based on surface roughness
    test_anopts['calc']           = calc   # how to evaluate fit parameters from simluation data - an_opts are 'Mean' and 'Median'
    test_anopts['calc_deff']     = calc_deff

    return test_bolo, test_anopts

def GS_test(test_fit, dsub, dSiOx, lw=1, ll=220, w1w=5, w2w=3):

    # returns G_S output from G_leg function for testing

    bolo0, anopts  = test_objs(              lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=0, dI1=0, dW2=0, dI2=0)
    boloI, anoptsI = test_objs(stack_I=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=0, dI1=0, dW2=0, dI2=0)
    boloN, anoptsN = test_objs(stack_N=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=0, dI1=0, dW2=0, dI2=0)

    GS0_bt  = G_bolotest(test_fit, anopts,  bolo0, layer='S')[0]/4
    GSI_bt  = G_bolotest(test_fit, anoptsI, boloI, layer='S')[0]/4
    GSN_bt  = G_bolotest(test_fit, anoptsN, boloN, layer='S')[0]/4

    GS0_leg  = G_leg(test_fit, anopts,  bolo0, dsub, 0, 0, 0, 0, True, False, False, legA=True)
    GSI_leg  = G_leg(test_fit, anoptsI, boloI, dsub, 0, 0, 0, 0, True, False, False, legA=True)
    GSN_leg  = G_leg(test_fit, anoptsN, boloN, dsub, 0, 0, 0, 0, True, False, False, legA=True)

    return np.array([GS0_bt, GSI_bt, GSN_bt]), np.array([GS0_leg, GSI_leg, GSN_leg])

def test_alphascaleS(verbose=False):

        lw     = 10;  ll = 220;      w1w = 10; w2w = 20
        dsub1  = 0.5; dsub2  = 1.0;  dsub3  = 2.0
        dSiOx1 = 0.1; dSiOx2 = 0.25; dSiOx3 = 0.5

        fit_a0  = np.array([1, 0, 1, 0, 0, 0])
        fit_aI1 = np.array([1, 0, 1, 0, 0, 1])
        fit_aS1 = np.array([1, 0, 1, 1, 0, 0])

        # all alpha = 0
        GS0a0_bt, GS0a0_leg = GS_test(fit_a0, dsub1, dSiOx1, lw=lw, w1w=w1w, w2w=w2w)
        GS0_a01_bt,  GSI_a01_bt,  GSN_a01_bt = GS0a0_bt; GS0_a01_leg, GSI_a01_leg, GSN_a01_leg = GS0a0_leg
        if verbose:
            # print('all alpha    = 0: GS_0 = {}; GS_I = {}; GS_N = {}'.format(round(GSleg0_a0_1, 1), round(GSlegI_a0_1, 1), round(GSlegN_a0_1, 1)))
            print('\n'); print('G_bolotest() output, all alpha = 0:')
            print('GS0  / GSN   = {}'.format(GS0_a01_bt/GSN_a01_bt))
            print('dsub / dSiOx = {}\n'.format(dsub1/dSiOx1))

            print('G_leg() output,      all alpha = 0:')
            print('GS0  / GSN   = {}'.format(GS0_a01_leg/GSN_a01_leg))
            print('dsub / dSiOx = {}\n'.format(dsub1/dSiOx1))

        assert all([G >= 0 for G in np.array([GS0a0_bt, GS0a0_leg]).flat]), "G < 0"
        assert round(GS0_a01_bt/GSN_a01_bt, 3) == dsub1/dSiOx1, "GS0_bt/GSN_bt != dsub/dSiOx"
        assert round(GS0_a01_leg/GSN_a01_leg, 3) == dsub1/dSiOx1, "GS0_leg/GSN_leg != dsub/dSiOx"

        # alpha_I = 1
        GS0aI1_bt, GS0aI1_leg = GS_test(fit_aI1, dsub1, dSiOx1, lw=lw, w1w=w1w, w2w=w2w)
        GS0_aI11_bt,  GSI_aI11_bt,  GSN_aI11_bt = GS0aI1_bt; GS0_aI11_leg, GSI_aI11_leg, GSN_aI11_leg = GS0aI1_leg
        if verbose:
            # print('all alpha    = 0: GS_0 = {}; GS_I = {}; GS_N = {}'.format(round(GSleg0_aI1_1, 1), round(GSlegI_aI1_1, 1), round(GSlegN_aI1_1, 1)))
            print('\n'); print('G_bolotest() output, aI = 1:')
            print('GS0  / GSN   = {}'.format(GS0_aI11_bt/GSN_aI11_bt))
            print('dsub / dSiOx = {}\n'.format(dsub1/dSiOx1))

            print('G_leg() output,      aI = 1:')
            print('GS0  / GSN   = {}'.format(GS0_aI11_leg/GSN_aI11_leg))
            print('dsub / dSiOx = {}\n'.format(dsub1/dSiOx1))

        assert all([G >= 0 for G in np.array([GS0aI1_bt, GS0aI1_leg]).flat]), "G < 0"
        assert round(GS0_aI11_bt/GSN_aI11_bt, 3)   == (dsub1/dSiOx1), "GS0_bt/GSN_bt   != (dsub1/dSiOx1)"
        assert round(GS0_aI11_leg/GSN_aI11_leg, 3) == (dsub1/dSiOx1), "GS0_leg/GSN_leg != (dsub1/dSiOx1)"

        # alpha_S = 1
        GS0aS1_bt, GS0aS1_leg = GS_test(fit_aS1, dsub1, dSiOx1, lw=lw, w1w=w1w, w2w=w2w)
        GS0_aS11_bt,  GSI_aS11_bt,  GSN_aS11_bt = GS0aS1_bt; GS0_aS11_leg, GSI_aS11_leg, GSN_aS11_leg = GS0aS1_leg
        if verbose:
            # print('all alpha    = 0: GS_0 = {}; GS_I = {}; GS_N = {}'.format(round(GSleg0_aS1_1, 1), round(GSlegI_aS1_1, 1), round(GSlegN_aS1_1, 1)))
            print('\n'); print('G_bolotest() output, aS = 1:')
            print('GS0  / GSN   = {}'.format(GS0_aS11_bt/GSN_aS11_bt))
            print('dsub / dSiOx = {}\n'.format(dsub1/dSiOx1))

            print('G_leg() output,      aS = 1:')
            print('GS0  / GSN   = {}'.format(GS0_aS11_leg/GSN_aS11_leg))
            print('dsub / dSiOx = {}\n'.format(dsub1/dSiOx1))

        assert all([G >= 0 for G in np.array([GS0aS1_bt, GS0aS1_leg]).flat]), "G < 0"
        assert round(GS0_aS11_bt/GSN_aS11_bt, 3)   == (dsub1/dSiOx1)**2, "GS0_bt/GSN_bt   != (dsub1/dSiOx1)^2"
        assert round(GS0_aS11_leg/GSN_aS11_leg, 3) == (dsub1/dSiOx1)**2, "GS0_leg/GSN_leg != (dsub1/dSiOx1)^2"

def GI_test(test_fit, dI1, dI2, lw=1, ll=220, w1w=5, w2w=3):

    # returns G_S output from G_leg function for testing

    bolo0, anopts  = test_objs(              lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=0, dSiOx=0, dW1=0, dI1=dI1, dW2=0, dI2=dI2)
    boloI, anoptsI = test_objs(stack_I=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=0, dSiOx=0, dW1=0, dI1=dI1, dW2=0, dI2=dI2)
    boloN, anoptsN = test_objs(stack_N=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=0, dSiOx=0, dW1=0, dI1=dI1, dW2=0, dI2=dI2)

    GI0_bt  = G_bolotest(test_fit, anopts,  bolo0, layer='I')[0]/4
    GII_bt  = G_bolotest(test_fit, anoptsI, boloI, layer='I')[0]/4
    GIN_bt  = G_bolotest(test_fit, anoptsN, boloN, layer='I')[0]/4

    GI0_leg  = G_leg(test_fit, anopts,  bolo0, 0, 0, dI1, 0, dI2, False, False, True, legA=True)
    GII_leg  = G_leg(test_fit, anoptsI, boloI, 0, 0, dI1, 0, dI2, False, False, True, legA=True)
    GIN_leg  = G_leg(test_fit, anoptsN, boloN, 0, 0, dI1, 0, dI2, False, False, True, legA=True)

    return np.array([GI0_bt, GII_bt, GIN_bt]), np.array([GI0_leg, GII_leg, GIN_leg])

def test_alphascaleI(verbose=False):

        dI1 = dI2 = 1.; dIstack = dI1 + dI2
        w1w = 5; w2w = 3; lw = 7

        fit_aI1  = np.array([0, 0, 1, 0, 0, 1])

        # w < w2 - GN = GI^2 = G0^2
        GI_w2_bt,  GI_w2_leg = GI_test(fit_aI1, dI1, dI2, lw=w2w, w1w=w1w, w2w=w2w)
        GI0_w2_bt, GII_w2_bt, GIN_w2_bt = GI_w2_bt; GI0_w2_leg, GII_w2_leg, GIN_w2_leg = GI_w2_leg
        if verbose:
            print('\n'); print('G_bolotest() output, w < w2w:')
            print('GIN  / GI0   = {}\n'.format(GIN_w2_bt/GI0_w2_bt))

            print('G_leg() output,      w < w2w:')
            print('GIN  / GI0   = {}\n'.format(GIN_w2_leg/GI0_w2_leg))
            # print('(dI1 + dI2)^2 / (dI1 + dI2) = {}\n'.format(dIstack**2/dIstack))
        assert all([G >= 0 for G in np.array([GI_w2_bt, GI_w2_leg]).flat]), "G < 0"
        assert round(GIN_w2_bt/GI0_w2_bt, 3)   == 1, "GIN_bt/GI0_bt   = {} != 1 for w < w2w".format(round(GIN_w2_bt/GI0_w2_bt, 3))
        assert round(GIN_w2_bt/GII_w2_bt, 3)   == 1, "GIN_bt/GII_bt   = {} != 1 for w < w2w".format(round(GIN_w2_bt/GII_w2_bt, 3))
        assert round(GIN_w2_leg/GI0_w2_leg, 3) == 1, "GIN_leg/GI0_leg = {} != 1 for w < w2w".format(round(GIN_w2_leg/GI0_w2_leg, 3))
        assert round(GIN_w2_leg/GII_w2_leg, 3) == 1, "GIN_leg/GII_leg = {} != 1 for w < w2w".format(round(GIN_w2_leg/GII_w2_leg, 3))

        # w2 < w < w1 - GN = GI = G0^2
        GI_w1_bt,   GI_w1_leg = GI_test(fit_aI1, dI1, dI2, lw=w1w, w1w=w1w, w2w=w2w)
        GI0_w1_bt,  GII_w1_bt,  GIN_w1_bt  = np.array(GI_w1_bt) - np.array(GI_w2_bt)
        GI0_w1_leg, GII_w1_leg, GIN_w1_leg = GI_w1_leg - GI_w2_leg
        if verbose:
            print('\n'); print('G_bolotest() output, w2w < w < w1w:')
            print('GIN  / GI0                  = {}'.format(round(GIN_w1_bt/GI0_w1_bt, 3)))
            print('(dI1 + dI2)^2 / (dI1 + dI2) = {}\n'.format(round(dIstack**2/dIstack, 3)))

            print('G_leg() output,      w2w < w < w1w:')
            print('GIN  / GI0                  = {}'.format(round(GIN_w1_leg/GI0_w1_leg, 3)))
            print('(dI1 + dI2)^2 / (dI1 + dI2) = {}\n'.format(round(dIstack**2/dIstack, 3)))
        assert all([G >= 0 for G in np.array([GI_w1_bt, GI_w1_leg]).flat]), "G < 0"
        assert round(GIN_w1_bt/GI0_w1_bt, 3)   == dIstack, "GIN_bt/GI0_bt   != (dI1 + dI2)^2 / (dI1 + dI2) for w2w < w < w1w"
        assert round(GIN_w1_bt/GII_w1_bt, 3)   == 1,       "GIN_bt/GII_bt   != 1f or w2w < w < w1w for w2w < w < w1w"
        assert round(GIN_w1_leg/GI0_w1_leg, 3) == dIstack, "GIN_leg/GI0_leg != (dI1 + dI2)^2 / (dI1 + dI2) for w2w < w < w1w"
        assert round(GIN_w1_leg/GII_w1_leg, 3) == 1,       "GIN_leg/GII_leg != 1 for w2w < w < w1w"

        # w > w1 - GN = GI = G0
        GI_lw_bt,   GI_lw_leg = GI_test(fit_aI1, dI1, dI2, lw=lw, w1w=w1w, w2w=w2w)
        GI0_lw_bt,  GII_lw_bt,  GIN_lw_bt  = GI_lw_bt  - GI_w1_bt  - GI_w2_bt
        GI0_lw_leg, GII_lw_leg, GIN_lw_leg = GI_lw_leg - GI_w1_leg - GI_w2_leg
        if verbose:
            print('\n'); print('G_bolotest() output, w > w1w:')
            print('GIN  / GI0   = {}'.format(GIN_lw_bt/GI0_lw_bt))

            print('G_leg() output,      w > w1w:')
            print('GIN  / GI0   = {}'.format(GIN_lw_leg/GI0_lw_leg))
        assert all([G >= 0 for G in np.array([GI_lw_bt, GI_lw_leg]).flat]), "G < 0"
        assert (round(GIN_lw_bt/GI0_lw_bt, 3)  ==1 and GIN_lw_bt>=0),  "GIN_bt/GI0_bt   != 1 for w > w1w"
        assert (round(GIN_lw_bt/GII_lw_bt, 3)  ==1 and GIN_lw_bt>=0),  "GIN_bt/GII_bt   != 1 for w > w1w"
        assert (round(GIN_lw_leg/GI0_lw_leg, 3)==1 and GIN_lw_leg>=0), "GIN_leg/GI0_leg != 1 for w > w1w"
        assert (round(GIN_lw_leg/GII_lw_leg, 3)==1 and GIN_lw_leg>=0), "GIN_leg/GII_leg != 1 for w > w1w"

def compare_output(fit, Gmeas=np.nan, lw=5, ll=220, dsub=0.400, dSiOx=0.120, w1w=5, w2w=3, dW1=0.200, dI1=0.400, dW2=0.350, dI2=0.400,
                    manual_calc = False, plot_vwidth=False, lwrange=np.arange(5,40), plot_vdsub=False, dsrange=np.arange(0.400, 2.500),
                    legA=False, legB=False, legC=False, legD=False, legE=False, legF=False, legG=False,
                    verbose=False, tall_Istacks=False, d_stacks=np.array([0, 0, 0, 0]), w_stacks=np.array([0, 0, 0, 0, 0]), deltalw_CD=0.0, extend_I2=False, calc_deff=False):
    # compare output between I-layer stacking at w > w1w ("no stacking / status quo / 0"), I-layer stacking at w > w2w, and nitride stacking
    # can look at different outputs for different legs vs width or thickness
    # manual_calc currently only works for leg A

    if manual_calc:   # calculate G manually and compare with function output
        # written to work for leg A

        bolo0, anopts0 = test_objs(              lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2)
        boloI, anoptsI = test_objs(stack_I=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2)
        boloN, anoptsN = test_objs(stack_N=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2)

        # pdb.set_trace()
        GS00 = fit[0]*(dsub/0.400)**(fit[3]+1)*lw/5
        GS0  = G_leg(fit, anopts0, bolo0, dsub, dW1, dI1, dW2, dI2, True, False, False, legA=True)
        GW0  = G_leg(fit, anopts0, bolo0, dsub, dW1, dI1, dW2, dI2, False, True, False, legA=True)
        GI0  = G_leg(fit, anopts0, bolo0, dsub, dW1, dI1, dW2, dI2, False, False, True, legA=True)
        G0   = G_leg(fit, anopts0, bolo0, dsub, dW1, dI1, dW2, dI2, True, True, True,   legA=True)
        if verbose: print('G_S no stack estimate difference: {} %\n'.format(round((GS0-GS00)/GS00*100, 1)))

        assert all([G >= 0 for G in np.array([GS00, GS0, GW0, GI0, G0]).flat]), "G < 0"
        assert (round(GS0/GS00, 3) == 1 or GS00==0), "GS0_leg/GS0_calc != 1"

        GSI0 = fit[0]*(dsub/0.400)**(fit[3]+1)*lw/5
        GSI  = G_leg(fit, anoptsI, boloI, dsub, dW1, dI1, dW2, dI2, True, False, False, legA=True)
        GWI  = G_leg(fit, anoptsI, boloI, dsub, dW1, dI1, dW2, dI2, False, True, False, legA=True)
        GII  = G_leg(fit, anoptsI, boloI, dsub, dW1, dI1, dW2, dI2, False, False, True, legA=True)
        GI   = G_leg(fit, anoptsI, boloI, dsub, dW1, dI1, dW2, dI2, True, True, True,   legA=True)
        if verbose: print('G_S I-layer stack estimate difference: {} %\n'.format(round((GSI-GSI0)/GSI0*100, 1)))
        assert (round(GSI/GSI0, 3) == 1 or GSI0==0), "GSI_leg/GSI_calc != 1"

        GSN0 = fit[0] * (dSiOx/0.400)**(fit[3]+1) * lw/5

        # initial nitride stack estimate
        dSiNx = dsub-dSiOx
        GI1N0_w2w = fit[2]*((dI1)/0.400)          **(fit[5]+1)*w2w/5   # I layer G out to 3 um
        GI2N0_w2w = fit[2]*((dI2)/0.400)          **(fit[5]+1)*w2w/5   # I layer G out to 3 um
        GIN0_w1w  = fit[2]*((dI1+dI2)/0.400)      **(fit[5]+1)*(w1w-w2w)/5 + GI1N0_w2w + GI2N0_w2w   # I layer G out to 5 um
        GSiNN0    = fit[2]*(dSiNx/0.400)          **(fit[5]+1)*w1w/5
        GSiNIN0   = fit[2]*((dSiNx+dI1+dI2)/0.400)**(fit[5]+1)*(lw-w1w)/5
        GIN0      = GSiNIN0 + GSiNN0 + GIN0_w1w

        # function output
        GSN     = G_leg(fit, anoptsN, boloN, dsub, dW1, dI1, dW2, dI2, True,  False, False, legA=True)
        GWN     = G_leg(fit, anoptsN, boloN, dsub, dW1, dI1, dW2, dI2, False, True,  False, legA=True)
        GIN     = G_leg(fit, anoptsN, boloN, dsub, dW1, dI1, dW2, dI2, False, False, True,  legA=True)
        GN      = G_leg(fit, anoptsN, boloN, dsub, dW1, dI1, dW2, dI2, True,  True,  True,  legA=True)

        # bolotest function output (for testing Leg A)
        GSN_bt  = G_bolotest(fit, anoptsN, boloN, layer='S')[0]/4
        GWN_bt  = G_bolotest(fit, anoptsN, boloN, layer='W')[0]/4
        GIN_bt  = G_bolotest(fit, anoptsN, boloN, layer='I')[0]/4
        GN_bt   = G_bolotest(fit, anoptsN, boloN, layer='total')[0]/4

        # just S nitride
        boloSiNx, anoptsSiNx = test_objs(stack_N=True, lw=lw, ll=ll, dI1=0, dI2=0, dW1=0, dW2=0, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx)
        GSiNx0   = fit[2] * (dSiNx/0.400)**(fit[5]+1) * lw/5
        GSiNx_bt = G_bolotest(fit, anoptsSiNx, boloSiNx, layer='I')[0]/4
        GSiNx    = G_leg(fit, anoptsSiNx, boloSiNx, dsub, 0, 0, 0, 0, 0, 0, 1, legA=True)

        if verbose:
            print('G_SiOx nitride stack estimate difference = {} %'.format(round(GSN-GSN0)/GSN0*100, 1))
            print('G_I    nitride stack estimate difference = {} %'.format(round(GIN-GIN0)/GIN0*100, 1))
            print('G_SiNx nitride stack estimate difference = {} %'.format(round(GSiNx-GSiNx0)/GSiNx0*100, 1))
            print('G_SiNx nitride is   = {} % of GI'.format(round(GSiNx/GIN*100, 1)))
            print('SiNx   nitride d is = {} % of total nitride d\n'.format(round(dSiNx/(dSiNx+dI1+dI2)*100, 1)))

            print('G_S    nitride stack bolotest output difference = {} %'.format(round(GSN_bt-GSN0)/GSN0*100, 1))
            print('G_SiNx nitride stack bolotest output difference = {} %'.format(round(GSiNx_bt-GSiNx0)/GSiNx0*100, 1))
            print('G_I    nitride stack bolotest output difference = {} %\n'.format(round(GIN_bt-GIN0)/GIN0*100, 1))

            print('G(I Stacks) > G(No Stacks) = {}'.format(GI > G0))
            print('G(N Stacks) > G(No Stacks) = {}'.format(GN > G0))
            print('G(N Stacks) > G(I Stacks)  = {}'.format(GN > GI))

        assert (round(GSN/GSN0, 3)        == 1 or GSN0==0),   "GSN_leg/GSN_calc     = {} != 1".format(round(GSN/GSN0, 3))
        assert (round(GSN_bt/GSN0, 3)     == 1 or GSN0==0),   "GS_bt/GS_calc        = {} != 1".format(round(GSN_bt/GSN0, 3))
        assert (round(GSiNx_bt/GSiNx0, 3) == 1 or GSiNx0==0), "GSiNx_bt/GSiNx_calc  = {} != 1".format(round(GSiNx_bt/GSiNx0, 3))
        assert (round(GSiNx/GSiNx0, 3)    == 1 or GSiNx0==0), "GSiNx_leg/GSiNx_calc = {} != 1".format(round(GSiNx/GSiNx0, 3))
        assert (round(GSiNx_bt/GSiNx0, 3) == 1 or GSiNx0==0), "GSiNx_bt/GSiNx_calc  = {} != 1".format(round(GSiNx_bt/GSiNx0, 3))
        assert (round(GIN/GIN0, 3)        == 1 or GIN0==0),   "GIN_leg/GIN_calc     = {} != 1".format(round(GIN/GIN0, 3))
        assert (round(GIN_bt/GIN0, 3)     == 1 or GIN0==0),   "GIN_bt/GIN_calc      = {} != 1".format(round(GIN_bt/GIN0, 3))
        print('\nG_leg() and G_bolotest() output matches manual calculations \n')

    if plot_vwidth:

        delta_lw = deltalw_CD if legC or legD else 0   # leg A and D seem to have a different width than the others

        bolo0_lw, anopts0_lw = test_objs(lw=lwrange, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2, w_stacks=w_stacks, d_stacks=d_stacks, tall_Istacks=tall_Istacks, extend_I2=extend_I2, calc_deff=calc_deff)
        G0_lw  = G_leg(fit, anopts0_lw, bolo0_lw, dsub, dW1, dI1, dW2, dI2, True, True, True,   legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        G0S_lw = G_leg(fit, anopts0_lw, bolo0_lw, dsub, dW1, dI1, dW2, dI2, True, False, False, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        G0W_lw = G_leg(fit, anopts0_lw, bolo0_lw, dsub, dW1, dI1, dW2, dI2, False, True, False, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        G0I_lw = G_leg(fit, anopts0_lw, bolo0_lw, dsub, dW1, dI1, dW2, dI2, False, False, True, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        assert all([G >= 0 for G in np.array([G0_lw, G0S_lw, G0W_lw, G0I_lw]).flat]), "G < 0"

        boloI_lw, anoptsI_lw = test_objs(stack_I=True, lw=lwrange, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2, w_stacks=w_stacks, d_stacks=d_stacks, tall_Istacks=tall_Istacks, extend_I2=extend_I2, calc_deff=calc_deff)
        GI_lw  = G_leg(fit, anoptsI_lw, boloI_lw, dsub, dW1, dI1, dW2, dI2, True, True, True,   legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        GIS_lw = G_leg(fit, anoptsI_lw, boloI_lw, dsub, dW1, dI1, dW2, dI2, True, False, False, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        GIW_lw = G_leg(fit, anoptsI_lw, boloI_lw, dsub, dW1, dI1, dW2, dI2, False, True, False, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        GII_lw = G_leg(fit, anoptsI_lw, boloI_lw, dsub, dW1, dI1, dW2, dI2, False, False, True, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        assert all([G >= 0 for G in np.array([GI_lw, GIS_lw, GIW_lw, GII_lw]).flat]), "G < 0"

        boloN_lw, anoptsN_lw = test_objs(stack_N=True, lw=lwrange, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2, w_stacks=w_stacks, d_stacks=d_stacks, tall_Istacks=tall_Istacks, extend_I2=extend_I2, calc_deff=calc_deff)
        GN_lw  = G_leg(fit, anoptsN_lw, boloN_lw, dsub, dW1, dI1, dW2, dI2, True, True, True,   legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        GNS_lw = G_leg(fit, anoptsN_lw, boloN_lw, dsub, dW1, dI1, dW2, dI2, True, False, False, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        GNW_lw = G_leg(fit, anoptsN_lw, boloN_lw, dsub, dW1, dI1, dW2, dI2, False, True, False, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        GNI_lw = G_leg(fit, anoptsN_lw, boloN_lw, dsub, dW1, dI1, dW2, dI2, False, False, True, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG, dI1I2=dI1+dI2)
        assert all([G >= 0 for G in np.array([GN_lw, GNS_lw, GNW_lw, GNI_lw]).flat]), "G < 0"

        # if legC: w2w = w1w; w1w = np.inf   # w2w is width of w1w on leg C, no w1w
        if legC: w1w = np.inf   # w2w is width of w1w on leg C, no w1w

        plt.figure(figsize=(10,5.5))
        plt.plot(lwrange/2, G0_lw,       alpha=0.8, linewidth=2.5, label='I Stacks $w>$W1')
        plt.plot(lwrange/2, GI_lw, '--', alpha=0.8, linewidth=2.5, label='I Stacks $w>$W2')
        plt.plot(lwrange/2, GN_lw, '-.', alpha=0.8, linewidth=2.5, label='N Stacks')
        plt.vlines([w2w/2, w1w/2, (lw+delta_lw)/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.7, color='k')
        if tall_Istacks:
            plt.vlines([np.array([w2w-w_stacks[0], w2w+w_stacks[1]])/2, np.array([w1w-w_stacks[2], w1w+w_stacks[3]])/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        elif extend_I2:
            plt.vlines(np.array([w2w+w_stacks[4], w2w+w_stacks[1]])/2, 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # plt.annotate('W2', (w2w/2-0.5, GN_lw[1]))
        # plt.annotate('W1', (w1w/2-0.5, GN_lw[1]))
        plt.annotate('W2', ((w2w-0.2)/2, np.max(GN_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        plt.annotate('W1', ((w1w-0.2)/2, np.max(GN_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        plt.annotate('leg', ((lw+delta_lw-0.15)/2,  np.max(GN_lw)*0.15), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        if np.isfinite(Gmeas):
            plt.hlines([Gmeas], 0, max(lwrange)/2, linestyle='-.', alpha=0.5, color='r', label='G$_{meas}$')
        plt.xlabel('1/2 Leg Width [$\mu m$]'); plt.ylabel('G [pW/K]')
        plt.ylim(0., np.nanmax([G0_lw, GI_lw, GN_lw])*1.1)
        plt.xlim(min(lwrange)/2, max(lwrange)/2)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)   # grid lines on plot
        plt.legend()

        # plt.figure(figsize=(10,5.5))
        # plt.plot(lwrange/2, G0W_lw,       alpha=0.8, linewidth=2.5, label='I Stacks $w>$W1')
        # plt.plot(lwrange/2, GIW_lw, '--', alpha=0.8, linewidth=2.5, label='I Stacks $w>$W2')
        # plt.plot(lwrange/2, GNW_lw, '-.', alpha=0.8, linewidth=2.5, label='N Stacks')
        # # plt.vlines([w2w/2, w1w/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.3, color='k')
        # plt.vlines([w2w/2, w1w/2, (lw+delta_lw)/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.7, color='k')
        # if tall_Istacks:
        #     plt.vlines([np.array([w2w-w_stacks[0], w2w+w_stacks[1]])/2, np.array([w1w-w_stacks[2], w1w+w_stacks[3]])/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # elif extend_I2:
            # plt.vlines(np.array([w2w+w_stacks[4], w2w+w_stacks[1]])/2, 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # # plt.annotate('W2', (w2w/2-0.3, GNW_lw[2]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # # plt.annotate('W1', (w1w/2-0.3, GNW_lw[2]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # plt.annotate('W2', ((w2w-0.2)/2, np.max(GNW_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.annotate('W1', ((w1w-0.2)/2, np.max(GNW_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.annotate('leg', ((lw+delta_lw-0.15)/2,  np.max(GNW_lw)*0.15), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.xlabel('1/2 Leg Width [$\mu m$]'); plt.ylabel('G$_W$ [pW/K]')
        # plt.ylim(min(GNW_lw)*0., np.nanmax([G0W_lw, GIW_lw, GNW_lw])*1.1)
        # plt.xlim(min(lwrange)/2, max(lwrange)/2)
        # plt.grid(linestyle = '--', which='both', linewidth=0.5)   # grid lines on plot
        # plt.legend()

        # plt.figure(figsize=(10,5.5))
        # plt.plot(lwrange/2, G0S_lw,       alpha=0.8, linewidth=2.5, label='I Stacks $w>$W1')
        # plt.plot(lwrange/2, GIS_lw, '--', alpha=0.8, linewidth=2.5, label='I Stacks $w>$W2')
        # plt.plot(lwrange/2, GNS_lw, '-.', alpha=0.8, linewidth=2.5, label='N Stacks')
        # # plt.vlines([w2w/2, w1w/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.3, color='k')
        # plt.vlines([w2w/2, w1w/2, (lw+delta_lw)/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.7, color='k')
        # if tall_Istacks:
        #     plt.vlines([np.array([w2w-w_stacks[0], w2w+w_stacks[1]])/2, np.array([w1w-w_stacks[2], w1w+w_stacks[3]])/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # elif extend_I2:
        #     plt.vlines(np.array([w2w+w_stacks[4], w2w+w_stacks[1]])/2, 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # # plt.annotate('W2', (w2w/2-0.3, GIS_lw[1]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # # plt.annotate('W1', (w1w/2-0.3, GIS_lw[1]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # plt.annotate('W2', ((w2w-0.2)/2, np.max(GIS_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.annotate('W1', ((w1w-0.2)/2, np.max(GIS_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.annotate('leg', ((lw+delta_lw-0.15)/2,  np.max(GIS_lw)*0.15), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.xlabel('1/2 Leg Width [$\mu m$]'); plt.ylabel('G$_S$ [pW/K]')
        # plt.ylim(0, np.nanmax([G0S_lw, GIS_lw, GNS_lw])*1.1)
        # # plt.ylim(0, 1.6)
        # plt.xlim(min(lwrange)/2, max(lwrange)/2)
        # plt.grid(linestyle = '--', which='both', linewidth=0.5)   # grid lines on plot
        # plt.legend()

        # params_test = np.array([0.507, 1.08, 1.66, -0.686, -0.384, -1.])

        plt.figure(figsize=(10,5.5))
        plt.plot(lwrange/2, G0I_lw,       alpha=0.8, linewidth=2.5, label='I Stacks $w>$W1')
        plt.plot(lwrange/2, GII_lw, '--', alpha=0.8, linewidth=2.5, label='I Stacks $w>$W2')
        plt.plot(lwrange/2, GNI_lw, '-.', alpha=0.8, linewidth=2.5, label='N Stacks')
        # plt.vlines([w2w/2, w1w/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.3, color='k')
        plt.vlines([w2w/2, w1w/2, (lw+delta_lw)/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.7, color='k')
        if tall_Istacks:
            plt.vlines([np.array([w2w-w_stacks[0], w2w+w_stacks[1]])/2, np.array([w1w-w_stacks[2], w1w+w_stacks[3]])/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        elif extend_I2:
            plt.vlines(np.array([w2w+w_stacks[4], w2w+w_stacks[1]])/2, 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # plt.annotate('W2', (w2w/2-0.3, GNI_lw[5]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # plt.annotate('W1', (w1w/2-0.3, GNI_lw[5]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        plt.annotate('W2', ((w2w-0.2)/2, np.max(GNI_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        plt.annotate('W1', ((w1w-0.2)/2, np.max(GNI_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        plt.annotate('leg', ((lw+delta_lw-0.15)/2,  np.max(GNI_lw)*0.15), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        plt.xlabel('1/2 Leg Width [$\mu m$]'); plt.ylabel('G$_I$ [pW/K]')
        plt.ylim(min(GNI_lw)*0., np.nanmax([G0I_lw, GII_lw, GNI_lw])*1.1)
        # plt.ylim(min(GNI_lw)*0., 14)
        plt.xlim(min(lwrange)/2, max(lwrange)/2)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)   # grid lines on plot
        plt.legend()

        # plt.figure(figsize=(10,5.5))
        # plt.plot(lwrange/2, G0S_lw + G0I_lw,       alpha=0.8, linewidth=2.5, label='I Stacks $w>$W1')
        # plt.plot(lwrange/2, GIS_lw + GII_lw, '--', alpha=0.8, linewidth=2.5, label='I Stacks $w>$W2')
        # plt.plot(lwrange/2, GNS_lw + GNI_lw, '-.', alpha=0.8, linewidth=2.5, label='N Stacks')
        # # plt.vlines([w2w/2, w1w/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.3, color='k')
        # plt.vlines([w2w/2, w1w/2, (lw+delta_lw)/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle='--', alpha=0.7, color='k')
        # if tall_Istacks:
        #     plt.vlines([np.array([w2w-w_stacks[0], w2w+w_stacks[1]])/2, np.array([w1w-w_stacks[2], w1w+w_stacks[3]])/2], 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # elif extend_I2:
            # plt.vlines(np.array([w2w+w_stacks[4], w2w+w_stacks[1]])/2, 0, np.nanmax([G0_lw, GI_lw, GN_lw])*1.1, linestyle=':', alpha=0.3, color='k')
        # # plt.annotate('W2', (w2w/2-0.3, GNI_lw[5]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # # plt.annotate('W1', (w1w/2-0.3, GNI_lw[5]))            # plt.xlabel('Leg Width [um]'); plt.ylabel('G [pW/K]')
        # plt.annotate('W2', ((w2w-0.2)/2, np.max(GNI_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.annotate('W1', ((w1w-0.2)/2, np.max(GNI_lw)*0.05), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.annotate('leg', ((lw+delta_lw-0.15)/2,  np.max(GNI_lw)*0.15), bbox=dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=1))
        # plt.xlabel('1/2 Leg Width [$\mu m$]'); plt.ylabel('G$_S$ + G$_I$ [pW/K]')
        # plt.ylim(min(GNS_lw+GNI_lw)*0., np.nanmax([G0S_lw+G0I_lw, GIS_lw+GII_lw, GNS_lw+GNI_lw])*1.1)
        # plt.xlim(min(lwrange)/2, max(lwrange)/2)
        # plt.grid(linestyle = '--', which='both', linewidth=0.5)   # grid lines on plot
        # plt.legend()

    if plot_vdsub:

        bolo0_ds, anopts0_ds = test_objs(lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2)
        G0_ds = G_leg(fit, anopts0_ds, bolo0_ds, dsrange, dW1, dI1, dW2, dI2, True, True, True, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG)

        boloI_ds, anoptsI_ds = test_objs(stack_I=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2)
        GI_ds = G_leg(fit, anoptsI_ds, boloI_ds, dsrange, dW1, dI1, dW2, dI2, True, True, True, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG)

        boloN_ds, anoptsN_ds = test_objs(stack_N=True, lw=lw, ll=ll, w1w=w1w, w2w=w2w, dsub=dsub, dSiOx=dSiOx, dW1=dW1, dI1=dI1, dW2=dW2, dI2=dI2)
        GN_ds = G_leg(fit, anoptsN_ds, boloN_ds, dsrange, dW1, dI1, dW2, dI2, True, True, True, legA=legA, legB=legB, legC=legC, legD=legD, legE=legE, legF=legF, legG=legG)

        plt.figure(figsize=(7,6))
        plt.plot(dsrange, G0_ds,       alpha=0.7, linewidth=2.5, label='I Stacks $w>$W1')
        plt.plot(dsrange, GI_ds, '--', alpha=0.7, linewidth=2.5, label='I Stacks $w>$W2')
        plt.plot(dsrange, GN_ds, '-.', alpha=0.7, linewidth=2.5, label='N Stacks')
        if np.isfinite(Gmeas):
            plt.vlines([dsub],  0, np.nanmax([G0_ds, GI_ds, GN_ds])*1.1, linestyle='--', alpha=0.5, color='k', label='d$_{sub}$')
            plt.hlines([Gmeas], 0, max(dsrange), linestyle='-.', alpha=0.5, color='r', label='G$_{meas}$')
        plt.xlabel('Substrate Thickness [um]'); plt.ylabel('G [pW/K]')
        plt.xlim(min(dsrange), max(dsrange))
        plt.ylim(0, np.nanmax([G0_ds, GI_ds, GN_ds]))
        plt.grid(linestyle = '--', which='both', linewidth=0.5)   # grid lines on plot
        plt.legend()

def test_widths(dsub=0.380, dSiOx=0.120, lw=7, w1w=6, w2w=4, dI1=0.5, dI2=0.6, stack_I=False, stack_N=False):

    # test separation of the leg into various regions

    # w_stacks =         [W2 slope, W2 edge, W1 slope, W1 edge, I2 extension]
    w_stacks1 = np.array([0.1,      0.2,     0.3,      0.35,    0.125])
    # d_stacks =        [deltad_AW2, deltad_AW1, deltad_CW2, deltad_DW1]
    d_stacks  = np.array([0.1,       0.2,        0.15,       0.25])

    bolo0, anopts0 = test_objs(                   stack_I=stack_I, stack_N=stack_N, lw=lw, dsub=dsub, w1w=w1w, w2w=w2w, dI1=dI1, dI2=dI2, w_stacks=w_stacks1, d_stacks=d_stacks)   # no  I layer thickness adjustments at W layer edges
    bolo1, anopts1 = test_objs(tall_Istacks=True, stack_I=stack_I, stack_N=stack_N, lw=lw, dsub=dsub, w1w=w1w, w2w=w2w, dI1=dI1, dI2=dI2, w_stacks=w_stacks1, d_stacks=d_stacks)   # yes I layer thickness adjustments at W layer edges
    bolo2, anopts2 = test_objs(extend_I2   =True, stack_I=stack_I, stack_N=stack_N, lw=lw, dsub=dsub, w1w=w1w, w2w=w2w, dI1=dI1, dI2=dI2, w_stacks=w_stacks1, d_stacks=d_stacks)   # yes I layer thickness adjustments at W layer edges

    region_ws0 = lw_regions(bolo0, anopts0); lw0, w2w_ns0, w1w_ns0, w2w_s0, w1w_s0, w2w_e0, w1w_e0, w2w_tot0, w1w_tot0, wI2_ext0 = region_ws0
    region_ws1 = lw_regions(bolo1, anopts1); lw1, w2w_ns1, w1w_ns1, w2w_s1, w1w_s1, w2w_e1, w1w_e1, w2w_tot1, w1w_tot1, wI2_ext1 = region_ws1
    region_ws2 = lw_regions(bolo2, anopts2); lw2, w2w_ns2, w1w_ns2, w2w_s2, w1w_s2, w2w_e2, w1w_e2, w2w_tot2, w1w_tot2, wI2_ext2 = region_ws1

    assert lw0==lw, "lw != input lw"
    assert lw1==lw, "lw != input lw"
    assert lw2==lw, "lw != input lw"

    assert all([w >= 0 for w in np.array(region_ws0).flat]), "w < 0"
    assert all([w >= 0 for w in np.array(region_ws1).flat]), "w < 0"
    assert all([w >= 0 for w in np.array(region_ws2).flat]), "w < 0"

    assert w2w_ns1 == w2w_ns0 - w_stacks1[0], "W2 sloped region is not correctly separated from w2w"
    assert w2w_s1  == w2w_s0  + w_stacks1[0], "W2 sloped region is not correctly separated from w2w"
    assert w1w_ns1 == w1w_ns0 - w_stacks1[2], "W1 sloped region is not correctly separated from w1w"
    assert w1w_s1  == w1w_s0  + w_stacks1[2], "W1 sloped region is not correctly separated from w1w"

    assert w2w_tot0 == w2w, "w2w_tot = {} != w2w (= {})".format(w2w_tot0, w2w)# not correctly separated into sloped and non-sloped regions"
    assert w1w_tot0 == w1w, "w1w_tot = {} != w1w (= {})".format(w1w_tot0, w1w)# not correctly separated into sloped and non-sloped regions"
    assert w2w_tot1 == w2w, "w2w_tot = {} != w2w (= {})".format(w2w_tot1, w2w)# not correctly separated into sloped and non-sloped regions"
    assert w1w_tot1 == w1w, "w1w_tot = {} != w1w (= {})".format(w1w_tot1, w1w)# not correctly separated into sloped and non-sloped regions"
    assert w2w_tot2 == w2w, "w2w_tot = {} != w2w (= {})".format(w2w_tot, w2w)# not correctly separated into sloped and non-sloped regions"
    assert w1w_tot2 == w1w, "w1w_tot = {} != w1w (= {})".format(w1w_tot, w1w)# not correctly separated into sloped and non-sloped regions"

    assert w2w_ns1 + w2w_s1 == w2w, "w2w_ns + w2w_s = {} != w2w (= {})".format(w2w_ns1 + w2w_s1, w2w)# not correctly separated into sloped and non-sloped regions"
    assert w1w_ns1 + w1w_s1 == w1w, "w1w_ns + w1w_s = {} != w1w (= {})".format(w1w_ns1 + w1w_s1, w1w)# not correctly separated into sloped and non-sloped regions"

    ### test taller I stacks lead to large G values
    test_fit = np.array([1, 1, 1, 0, 0, 1])   # test fit parameters
    G0_legA  = G_leg(     test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legA=True)
    G0_bt    = G_bolotest(test_fit, anopts0, bolo0, layer='I')[0]/4
    G1_legA  = G_leg(     test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legA=True)
    G1_bt    = G_bolotest(test_fit, anopts1, bolo1, layer='I')[0]/4
    G2_legA  = G_leg(     test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legA=True)
    G2_bt    = G_bolotest(test_fit, anopts2, bolo2, layer='I')[0]/4

    assert G0_legA == G0_bt,  "G_legA = {} != G_bt = {}".format(round(G0_legA, 2), round(G0_bt, 2))
    assert G1_legA == G1_bt,  "G_legA = {} != G_bt = {}".format(round(G1_legA, 2), round(G1_bt, 2))
    assert G2_legA == G2_bt,  "G_legA = {} != G_bt = {}".format(round(G1_legA, 2), round(G1_bt, 2))
    assert G1_legA > G0_legA, "G_legA(taller I stacks) = {} !> nominal G_legA = {}".format(round(G1_legA, 2), round(G0_legA, 2))
    assert G2_legA > G0_legA, "G_legA(extend I2)       = {} !> nominal G_legA = {}".format(round(G2_legA, 2), round(G0_legA, 2))

    ### test other leg geometries
    # leg B should be the same regardless of I stack treatement
    G0_legB = G_leg( test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legB=True)
    G1_legB = G_leg( test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legB=True)
    G2_legB = G_leg( test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legB=True)
    assert G0_legB == G1_legB, "nominal G_legB = {} != I stacking  G_legB = {}".format(round(G0_legB, 2), round(G1_legB, 2))
    assert G0_legB == G2_legB, "nominal G_legB = {} != extended I2 G_legB = {}".format(round(G0_legB, 2), round(G2_legB, 2))

    # leg C with taller I stacks should be larger than nominal
    G0_legC = G_leg( test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legC=True)
    G1_legC = G_leg( test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legC=True)
    G2_legC = G_leg( test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legC=True)
    assert G1_legC > G0_legC,  "G_legC(taller I stacks) = {} !> nominal G_legC = {}".format(round(G1_legC, 2), round(G0_legC, 2))
    assert G2_legC > G0_legC,  "G_legC(extended I2)     = {} !> nominal G_legC = {}".format(round(G2_legC, 2), round(G0_legC, 2))

    # leg D with taller I stacks should be larger than nominal
    G0_legD = G_leg( test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legD=True)
    G1_legD = G_leg( test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legD=True)
    G2_legD = G_leg( test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legD=True)
    assert G1_legD >  G0_legD, "G_legD(taller I stacks) = {} !> nominal G_legD = {}".format(round(G1_legD, 2), round(G0_legD, 2))
    assert G0_legD == G2_legD, "nominal G_legD = {} != extended I2 G_legD = {}".format(round(G0_legD, 2), round(G2_legD, 2))

    # leg E should be the same regardless of I stack treatement
    G0_legE = G_leg( test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legE=True)
    G1_legE = G_leg( test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legE=True)
    G2_legE = G_leg( test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legE=True)
    assert G0_legE == G1_legE, "nominal G_legE = {} != I stacking G_legE = {}".format(round(G0_legE, 2), round(G1_legE, 2))
    assert G0_legE == G2_legE, "nominal G_legE = {} != extended I2 G_legE = {}".format(round(G0_legE, 2), round(G2_legE, 2))

    # leg F should be the same regardless of I stack treatement
    G0_legF = G_leg( test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legF=True)
    G1_legF = G_leg( test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legF=True)
    G2_legF = G_leg( test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legF=True)
    assert G0_legF == G1_legF, "nominal G_legF = {} != I stacking G_legF = {}".format(round(G0_legF, 2), round(G1_legF, 2))
    assert G0_legF == G2_legF, "nominal G_legF = {} != extended I2 G_legF = {}".format(round(G0_legF, 2), round(G2_legF, 2))

    # leg G should be the same regardless of I stack treatement
    G0_legG = G_leg( test_fit, anopts0, bolo0, dsub, 0, dI1, 0, dI2, False, False, True, legG=True)
    G1_legG = G_leg( test_fit, anopts1, bolo1, dsub, 0, dI1, 0, dI2, False, False, True, legG=True)
    G2_legG = G_leg( test_fit, anopts2, bolo2, dsub, 0, dI1, 0, dI2, False, False, True, legG=True)
    assert G0_legG == G1_legG, "nominal G_legG = {} != I stacking G_legG = {}".format(round(G0_legG, 2), round(G1_legG, 2))
    assert G0_legG == G2_legG, "nominal G_legG = {} != extended I2 G_legG = {}".format(round(G0_legG, 2), round(G2_legG, 2))

    return


def test_deff(alpha=1, left_frac=1., plot_deff=False):

    d   = 1
    w   = 2*d
    mfp = 1/(1/d + 1/w)

    deff_calc = (d*mfp**alpha)**(1/(1+alpha))

    fit = np.array([1, 1, 1, 1, 1, alpha])
    deff_func = deff(fit, w, d)

    out_ratio        = round(deff_func/deff_calc, 3)
    assert out_ratio == 1, "deff(function) / deff(hand calc) = {} != 1".format(out_ratio)

    # A, C, and D are the same if:
    # widths of regions a and b = 0
    # dW1 = dI1 = dI1I2/2
    d_ACD = 2; dI1I2_ACD = 2*d_ACD
    lw = 6

    bolo_ACD, anopts_ACD = test_objs(stack_I=True, w1w=1E-12, w2w=1E-12, lw=lw, dI1=d_ACD, dI2=d_ACD, dW1=d_ACD)

    mfpIcalc_2walls = 1/(1/dI1I2_ACD + 1/lw)
    mfpIcalc_1wall  = 1/(1/dI1I2_ACD + 1/(2*lw))
    deffI_calc = (dI1I2_ACD * mfpIcalc_2walls**alpha)**(1/(1+alpha))*0.5 + (dI1I2_ACD * mfpIcalc_1wall**alpha)**(1/(1+alpha))*0.5

    # leg A
    regionws_A   = lw_regions(bolo_ACD, anopts_ACD); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_A
    wI1I2nom_ACD = lw - (w2w_tot + w2w_e)
    dI1I2eff_A   = deff_I1I2(fit, regionws_A, wI1I2nom_ACD, dI1I2_ACD, d_ACD, d_ACD, legA=True, left_frac=left_frac)

    # leg C
    deltaw2w_C = bolo_ACD['geometry']['w1w'] - bolo_ACD['geometry']['w2w']
    regionws_C = lw_regions(bolo_ACD, anopts_ACD, delta_w2w=deltaw2w_C); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_C
    dI1I2eff_C = deff_I1I2(fit, regionws_C, wI1I2nom_ACD, dI1I2_ACD, d_ACD, d_ACD, legC=True, left_frac=left_frac)

    # leg D
    regionws_D = lw_regions(bolo_ACD, anopts_ACD); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_D
    dI1I2eff_D = deff_I1I2(fit, regionws_D, wI1I2nom_ACD, dI1I2_ACD, d_ACD, d_ACD, legD=True, left_frac=left_frac)

    # Avscalc = round(dI1I2eff_A/deffI_calc, 3)
    # assert Avscalc == 1, "deff_A / hand-calcluated deff = {} != 1".format(Avscalc)

    AoverC = round(dI1I2eff_A/dI1I2eff_C, 3)
    assert AoverC == 1, "deff_A / deff_C = {} != 1, but it should in this instance".format(AoverC)

    AoverD = round(dI1I2eff_A/dI1I2eff_D, 3)
    assert AoverD == 1, "deff_A / deff_D = {} != 1, but it should in this instance".format(AoverD)

    if plot_deff:
        dW10 = 0.160; dW20 = 0.350
        dI10 = 0.350; dI20 = 0.300
        lw0  = 6.0

        test_bolo, test_anopts = test_objs(stack_I=True, dW1=dW10, dW2=dW20, dI1=dI10, dI2=dI20, lw=lw0)

        deltalw_CD = test_bolo['geometry']['deltalw_CD']

        dI2_test = np.linspace(dI10, dI10+dW20)
        w1w = test_bolo['geometry']['w1w']; w2w = test_bolo['geometry']['w1w']; lw = test_bolo['geometry']['lw']

        dI2_eff  = np.array([deff(fit, w2w, dI) for dI in dI2_test])
        plt.figure()
        plt.plot(dI2_test*1E3, dI2_eff/dI2_test*100)
        plt.xlabel('dI2 [nm]')
        plt.ylabel('dI2_eff [\% of dI2]')
        plt.xlim(min(dI2_test)*1E3, max(dI2_test)*1E3)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)

        dW2_test = np.linspace(0.100, 0.400)
        dW2_eff  = np.array([deff(fit, w2w, dW, alphaind=4) for dW in dW2_test])
        plt.figure()
        plt.plot(dW2_test*1E3, dW2_eff/dW2_test*100)
        plt.xlabel('dW2 [nm]')
        plt.ylabel('dW2_eff [\% of dW2]')
        plt.xlim(min(dW2_test)*1E3, max(dW2_test)*1E3)
        plt.grid(linestyle = '--', which='both', linewidth=0.5)

        # plot dI1I2_eff
        dI1I2_test = np.linspace(dI10, 1.000)

        # leg A
        regionws_A = lw_regions(test_bolo, test_anopts); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_A
        wI1I2nom_A = lw - (w2w_tot + w2w_e)
        dI1I2eff_A = deff_I1I2(fit, regionws_A, wI1I2nom_A, dI1I2_test, dI10, dW10, legA=True, left_frac=left_frac)

        # leg D
        regionws_D = lw_regions(test_bolo, test_anopts, delta_lw=deltalw_CD); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_D
        wI1I2nom_D = lw - (w1w_s + w1w_e)
        dI1I2eff_D = deff_I1I2(fit, regionws_D, wI1I2nom_D, dI1I2_test, dI10, dW10, legD=True, left_frac=left_frac)

        # leg C
        regionws_C = lw_regions(test_bolo, test_anopts, delta_lw=deltalw_CD); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_C
        wI1I2nom_C = lw - (w2w_e + w2w_tot)
        dI1I2eff_C = deff_I1I2(fit, regionws_C, wI1I2nom_C, dI1I2_test, dI10, dW10, legC=True, left_frac=left_frac)

        plt.figure()
        plt.plot(dI1I2_test*1E3, dI1I2eff_A/dI1I2_test*100, label='leg A')
        plt.plot(dI1I2_test*1E3, dI1I2eff_C/dI1I2_test*100, label='leg C')
        plt.plot(dI1I2_test*1E3, dI1I2eff_D/dI1I2_test*100, label='leg D')
        plt.ylabel('dI1I2_eff [\% of dI1I2]')
        plt.xlabel('dI1I2 [nm]')
        plt.xlim(min(dI1I2_test)*1E3, max(dI1I2_test)*1E3)
        plt.legend()
        plt.grid(linestyle = '--', which='both', linewidth=0.5)

        dI1I20 = 0.600
        lw_test = np.linspace(w1w, 7.0)
        test_bolo, test_anopts = test_objs(stack_I=True, dW1=dW10, dW2=dW20, dI1=dI10, dI2=dI20, lw=lw_test)

        # leg A
        regionws_A = lw_regions(test_bolo, test_anopts); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_A
        wI1I2nom_A = lw - (w2w_tot + w2w_e)
        dI1I2eff_A = deff_I1I2(fit, regionws_A, wI1I2nom_A, dI1I20, dI10, dW10, legA=True, left_frac=left_frac)

        # leg D
        regionws_D = lw_regions(test_bolo, test_anopts, delta_lw=deltalw_CD); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_D
        wI1I2nom_D = lw - (w1w_s + w1w_e)
        dI1I2eff_D = deff_I1I2(fit, regionws_D, wI1I2nom_D, dI1I20, dI10, dW10, legD=True, left_frac=left_frac)

        # leg C
        regionws_C = lw_regions(test_bolo, test_anopts, delta_lw=deltalw_CD); lw, w2w_ns, w1w_ns, w2w_s, w1w_s, w2w_e, w1w_e, w2w_tot, w1w_tot, wI2_ext = regionws_C
        wI1I2nom_C = lw - (w2w_e + w2w_tot)
        dI1I2eff_C = deff_I1I2(fit, regionws_C, wI1I2nom_C, dI1I20, dI10, dW10, legC=True, left_frac=left_frac)

        plt.figure()
        plt.plot(lw_test, dI1I2eff_A/dI1I20*100, label='leg A')
        plt.plot(lw_test, dI1I2eff_C/dI1I20*100, label='leg C')
        plt.plot(lw_test, dI1I2eff_D/dI1I20*100, label='leg D')
        plt.ylabel('dI1I2_eff [\% of dI1I2]')
        plt.xlabel('lw [um]')
        plt.xlim(min(lw_test), max(lw_test))
        plt.legend()
        plt.grid(linestyle = '--', which='both', linewidth=0.5)

    return