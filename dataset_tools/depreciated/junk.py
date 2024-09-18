    def run_production_reverse_side(self,):

        multi = True # Specify if want 2d/3d etch-pit images

        """
        Get Clean etch-pit locations
        ------------------------------------------------------------------
        """

        location_file='./pit_locations.txt'
        key='Confirmed pits'
        pit_locations = 0
        with open(location_file) as json_file:
            pit_locations = json.load(json_file)
        pit_locations = pit_locations[key]

        # Set of empty slides needs to be slightly different 
        empty = [200,197,150,145,123,100,225] #,244,250]
        bkg_loc = self.get_background(empty)

        """
        Crop etch-pits on either side of the foil
        ------------------------------------------------------------------
        """
        def crop(pits,foil,offset): 
            """
            Applies a coordinate transformation and crops from specified foil
            """
            off = pit_alignment.Offset(offset)
            newpits = list(map(off.convert_ixy,pits))
            this_process = (lambda x : self.process(x,foil,multi))
            cropped_pits = list(map(this_process,newpits))
            return cropped_pits

        sig  = crop(pit_locations,'dirty_s'  ,'cd')
        sigr = crop(pit_locations,'dirty_r_s','cdr')
        bkg  = crop(bkg_loc,'dirty_s'  ,'cd')
        bkgr = crop(bkg_loc,'dirty_r_s','cdr')

        print(len(sig))
        print(len(sigr))
        print(len(bkg))
        print(len(bkgr))

        # Something is bugged, so will output now / early to manually check over
        pits = {
            'sig' : sig,
            'sigr' : sigr,
            'bkg' : bkg,
            'bkgr' : bkgr,
                }

        path_out = './pits/pits_frontback_2d/pits_3d'
        with open(path_out,'wb') as outfile:
            pickle.dump(pits,outfile)



    def rprs2(self,):

        """
        Check BOTH pits are within the correct margin, and cropped correctly   
        -------------------------------------------------------------------   
        """

        def check_fb(A,B):
            C = [(a,b) for a,b in zip(A,B) if (a != None and b != None)] 
            print('Initial # pits :{}	Successfuly cropped # :{}'.format(len(A),len(C)))
            return zip(*C)
        
        def check_fb2(A,B):
            print(B)
            for a,b in zip(A,B):
                if (a != None): print('a non zero')
                if (b != None): print('b non zero')
            C = []
            print('Initial # pits :{}	Successfuly cropped # :{}'.format(len(A),len(C)))
            return zip(*C)

        pits = 0
        with open('./pits/pits_frontback_2d/pits_3d','rb') as pickle_file:
            pits = pickle.load(pickle_file)

        sig = pits['sig']
        sigr = pits['sigr']
        bkg = pits['bkg']
        bkgr = pits['bkgr']

        print('Check signal')   
        sig,sigr = check_fb(sig,sigr)
        print('Check background') 
        bkg,bkgr = check_fb(bkg,bkgr)


        # Test a pit or two

        pit = sigr[0] # take 1st elemwnt in list
        pit = pit[3] # take the image
        it.plot(pit)

        pits = {
            'sig' : sig,
            'sigr' : sigr,
            'bkg' : bkg,
            'bkgr' : bkgr,
                }

        path_out = './pits/pits_frontback_2d/pits_3d_checked'
        with open(path_out,'wb') as outfile:
            pickle.dump(pits,outfile)

