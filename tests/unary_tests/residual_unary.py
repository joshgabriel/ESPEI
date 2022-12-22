from typing import Dict, List, Optional, Protocol, Tuple, Type

from numpy.typing import ArrayLike

from pycalphad import Database
from glob import glob
from espei.phase_models import PhaseModelSpecification
from espei.typing import SymbolName
from espei.utils import PickleableTinyDB
from tinydb.storages import MemoryStorage
from espei.error_functions.residual_base import ResidualFunction


class SGTE91UnaryResidual(ResidualFunction):
    """
    Class that defines SGTE91 Residual function for co-optimization
    of unaries and binaries. Inherits from ResidualFunction base class
    class with
    attributes: data, error, likelihood.
    methods: various heat capacity functions, m.p, H_fusion, etc.
    """
    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB
        #phase_models: PhaseModelSpecification,
        #symbols_to_fit: Optional[List[SymbolName]],
        #weight: Optional[Dict[str, float]]
        ):
            # parameters becomes symbols_to_fit, 
            # data_sources becomes datasets 
            # weight is a none for now
            # phase models: for unaries?
            self.database = database 
            self.data_db = datasets
            # populate self.data from datasets,database
            self.data = {k:{} for k in ["Cp","H","Hfus","Tm"]}
            # check how to unpack a PickleableTinyDB
            for json_data in self.data_db.all():
                #jsonf = open(jf) # access record of TinyDB into dict
                #json_data=json.load(jsonf) #
                self.data[json_data['type']].update({json_data['name']:json_data['data']})
                jsonf.close()

    def get_residuals(self, parameters: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        CALCULATES ERROR FUNCTIONS
        Currently parameters are attributes of the class
        manager for all property difference calculations
         say data is a dict:
         {"Cp":{"DatasetN":{"T":[],"Cp":[],"U":[],"P":[]}},
          "H":{"DatasetN":{"T":[],"H":[],"U":[],"P":[]}},
          "Hfus":{"DatasetN":{"T":[],"H":[],"U":[],"P":[]}}
          "Tm":{"DatasetN":{"T":[],"H":[],"U":[],"P":[]}}}

        Return the residual comparing the selected data to the set of parameters.

        The residual is zero if the database predictions under the given
        parameters agrees with the data exactly.

        Parameters
        ----------
        parameters : ArrayLike
            1D parameter vector. The size of the parameters array should match
            the number of fitting symbols used to build the models. This is
            _not_ checked.

        Returns
        -------
        Tuple[List[float], List[float]]
            Tuple of (residuals, weights), which must obey len(residuals) == len(weights)

        """
        ## get Cp data for solid
        # calculate Cp and compute error
        self.A, self.B, self.D, self.E, \
            self.Ssolid, self.Sliquid, self.H298, self.Tb, self.Tm_range = self.parameters
        self.Tm_low, self.Tm_high = Tm_range
        self.evaluate_continuity_params()
        self.evaluate_gibbs_S_params()
        self.evaluate_S_continuity_params()
        self.evaluate_last()

        Cp_error = []
        Cp_weights = []
        for k in self.data["Cp"]:
             temps = self.data["Cp"][k]["T"]
             dataset = np.array(self.data["Cp"][k]["Cp"])
             Cp_weights.append(np.array(self.data["Cp"][k]["U"]))
             if self.data["Cp"][k]["P"][0]=="solid":
                #print (self.Cp_solid(temps) - dataset)
                Cp_error.append(self.Cp_solid(temps)-dataset)
             elif self.data["Cp"][k]["P"][0]=="liquid":
                Cp_error.append(self.Cp_liquid(temps)-dataset)

        H_error = []
        H_weights = []
        for k in self.data["H"]:
             temps = self.data["H"][k]["T"]
             dataset = np.array(self.data["H"][k]["H"])
             #print ("H dataset",dataset,len(dataset),len(temps))
             H_weights.append(np.array(self.data["H"][k]["U"]))
             if self.data["H"][k]["P"][0]=="solid":
                #print ("solid",self.H_solid(temps),dataset,self.H_solid(temps)-dataset)
                H_error.append(self.H_solid(temps)-dataset)
             elif self.data["H"][k]["P"][0]=="liquid":
                #print ("liquid",self.H_liquid(temps),dataset,self.H_liquid(temps)-dataset)
                H_error.append(self.H_liquid(temps)-dataset)

        M_error = []
        M_weights = []
        self.Tm_solved(self.Tm_low,self.Tm_high)
        for k in self.data["Tm"]:
             dataset = np.array(self.data["Tm"][k]["Tm"])
             M_weights.append(np.array(self.data["Tm"][k]["U"]))
             M_error.append(self.Tm-dataset)

        Hfus_error = []
        Hfus_weights = []
        self.Hfus_calc()
        for k in self.data["Hfus"]:
             dataset = np.array(self.data["Hfus"][k]["Hfus"])
             Hfus_weights.append(np.array(self.data["Hfus"][k]["U"]))
             Hfus_error.append(self.H_fusion-dataset)

        weights = Cp_weights + H_weights + M_weights + Hfus_weights
        errors = Cp_error + H_error + M_error + Hfus_error
        #self.weights = Cp_weights + M_weights + Hfus_weights
        #self.errors = Cp_error + M_error + Hfus_error
        _log.trace("Unary Errors - %s", self.errors)
        _log.trace("Unary Weights - %s", self.weights)
        return weights, errors

    def get_likelihood(self, parameters) -> float:
        """
        Using t-distribution for likelihood;
        could be changed to normal distribution
        for consistency with rest of espei

        Return log-likelihood for the set of parameters.

        Parameters
        ----------
        parameters : ArrayLike
            1D parameter vector. The size of the parameters array should match
            the number of fitting symbols used to build the models. This is
            _not_ checked.

        Returns
        -------
        float
            Value of log-likelihood for the given set of parameters

        """
        self.errors = self.get_residuals(parameters)
        prob = 0
        dof = 2+1e-06
        _log.trace("Unary errors length %f %f",len(self.errors),len(self.weights))
        for num,e in enumerate(self.errors):
            #print (num,e,prob)
            prob+= ss.t.logpdf(e, dof, loc=0, scale=self.weights[num]).sum()
            _log.trace("Prob unary %f %f %f",num,e,prob)
        if prob == np.nan:
           prob = -np.inf
        _log.trace("Returning unary prob is %f", prob)
        return prob

    # Lots of helper functions
    def Cp_solid(self,T):
        """
        calculate Cp for solid for given temperature
        """
        T_low = np.array([t for t in T if t <= self.Tb])
        T_hi = np.array([t for t in T if t > self.Tb])
        return list(-T_low*(6*self.A*T_low + 2*self.B*T_low**0 + (self.C/T_low) \
               + 2*self.D*(T_low**(-3)))) + \
               list(-T_hi*((self.E/T_hi) + self.F*(90)*(T_hi**(-11))))

    def Cp_liquid(self,T):
        """
        calculate Cp for liquid for given temperature
        """
        T_low = np.array([t for t in T if t <= self.Tb])
        T_hi = np.array([t for t in T if t > self.Tb])
        return list(-T_low*(42*self.G*T_low**5 + 6*self.A*T_low + \
                    2*self.B*T_low**0 + \
                    self.C/T_low + 2*self.D*T_low**-3)) + \
               list(-self.E*T_hi**0)

    def evaluate_continuity_params(self):
        #900*F*T**-11 = -12*A*T -2*B*T**0 +6*D*T**(-3)
        self.F = (-12*self.A*self.Tb -2*self.B*self.Tb**0 +\
                  6*self.D*self.Tb**(-3))/(900*self.Tb**-11)

        # (6*A*T**2 + 2*B*T + C + 2*D*(T**(-2))) = E + F*(90)*(T**(-10))
        self.C = self.E + self.F*(90)*(self.Tb**(-10)) -6*self.A*self.Tb**2 -\
                 2*self.B*self.Tb -2*self.D*(self.Tb**(-2))

        self.G = (self.E - 6*self.A*self.Tb**2 - 2*self.B*self.Tb - self.C*self.Tb**0 - \
                 2*self.D*self.Tb**-2)/(42*self.Tb**6)
        #print (self.F,self.C,self.G)

    def evaluate_gibbs_S_params(self):
        T = 298.15
        self.b = -1*(3*self.A*T**2 + 2*self.B*T + self.C + self.C*np.log(T) -\
                 self.D*T**(-2)) - self.Ssolid
        T = self.Tb
        self.h =  -self.E - self.E*np.log(T) - self.Sliquid

    def evaluate_S_continuity_params(self):
        T = self.Tb
        self.d =  -self.S_low_s(T) - self.E - self.E*np.log(T) + 9*self.F*T**-10
        self.f = self.S_low_s(T) -7*self.G*T**6 + self.b -self.S_hi_l(T)

    def S_low_s(self,T):
        return -1*(self.b + 3*self.A*T**2 + 2*self.B*T + self.C + \
        self.C*np.log(T) - self.D*T**(-2))

    def S_hi_s(self,T):
        return -1*(self.d + self.E + self.E*np.log(T) -9*self.F*T**(-10))

    def S_low_l(self,T):
          return self.S_low_s(T) + -1*(7*self.G*T**6 + self.f - self.b)

    def S_hi_l(self,T):
          return -1*(self.h + self.E + self.E*np.log(T))

    def G_low_s_integrated(self,T_range,Tref):

        def Heval(T):
            return -2*self.A*T**3 - self.B*T**2 - self.C*T + 2*self.D*T**(-1)

        H_term = np.array([Heval(T) - Heval(Tref) for T in T_range])
        TS_term = np.array([-T*self.S_low_s(T) for T in T_range])
        G_eval = H_term + TS_term
        return G_eval

    def G_hi_s_integrated(self,T_range,Tref):

        def Heval(T):
            return -self.E*T + 10*self.F*T**-9

        def Heval_low(T):
            return -2*self.A*T**3 - self.B*T**2 - self.C*T + 2*self.D*T**(-1)

        H_term = np.array([Heval(T) - Heval(Tref) for T in T_range])
        H_ref_term = np.array([Heval_low(Tref) - Heval_low(298.15) for T in T_range])
        TS_term = np.array([-T*self.S_hi_s(T) for T in T_range])
        G_eval = H_ref_term + H_term + TS_term
        return G_eval

    def G_s_hser(self,T):
        if T<=self.Tb:
           return self.G_low_s_integrated([T],298.15)
        else:
           return self.G_hi_s_integrated([T],self.Tb)

    def G_low_l_integrated(self,T_range,Tref):

        def Heval(T):
            return -2*self.A*T**3 - self.B*T**2 - self.C*T + 2*self.D*T**(-1) \
            - 6*self.G*T**7

        H_term = np.array([Heval(T) - Heval(Tref) for T in T_range])
        TS_term = np.array([-T*self.S_low_l(T) for T in T_range])
        G_eval = np.array([self.H298 for T in T_range]) + H_term + TS_term

        return G_eval

    def G_hi_l_integrated(self,T_range,Tref):
        def Heval(T):
            return -self.E*T

        def Heval_low(T):
            return -2*self.A*T**3 - self.B*T**2 - self.C*T + 2*self.D*T**(-1)\
             - 6*self.G*T**7

        H_term = np.array([Heval(T) - Heval(Tref) for T in T_range])
        H_ref_term = np.array([Heval_low(Tref) - Heval_low(298.15) for T in T_range])
        TS_term = np.array([-T*self.S_hi_l(T) for T in T_range])
        G_eval = np.array([self.H298 for T in T_range]) + H_ref_term + H_term + TS_term
        return G_eval

    def G_l_hser(self,T):
        if T<=self.Tb:
           return self.G_low_l_integrated([T],298.15)
        else:
           return self.G_hi_l_integrated([T],self.Tb)

    def G_HSER_solid_liquid(self,T):
        G_solid_HSER = self.G_s_hser(T)
        G_liquid_HSER = self.G_l_hser(T)
        return G_solid_HSER - G_liquid_HSER

    def Tm_solved(self,Tsl,Tsh):
        try:
           Tm_r = ridder(self.G_HSER_solid_liquid,Tsl,Tsh)
           if abs(self.G_HSER_solid_liquid(Tm_r))>1e-3:
              self.Tm = np.nan
           self.Tm = Tm_r
           _log.trace("Tm solved %f",self.Tm)
        except ValueError:
           _log.trace("Tm not solved within %f %f",Tsl,Tsh)
           self.Tm = np.nan

    def Hfus_calc(self):
        if self.Tm > self.Tb:
           H_298_Tb_solid = self.int_Cp_s_low(298.15,self.Tb)
           H_Tb_Tm_solid = self.int_Cp_s_hi(self.Tb,self.Tm)
           H_Tm_solved_solid = H_298_Tb_solid + H_Tb_Tm_solid

           H_298_Tb_liquid = self.int_Cp_l_low(298.15,self.Tb)
           H_Tb_Tm_liquid = self.int_Cp_l_hi(self.Tb,self.Tm)
           H_Tm_solved_liquid = H_298_Tb_liquid + self.H298 + H_Tb_Tm_liquid
           self.H_fusion = H_Tm_solved_liquid - H_Tm_solved_solid

        else:
           H_Tm_solved_solid = self.int_Cp_s_low(298.15,self.Tm)
           H_Tm_solved_liquid = self.int_Cp_l_low(298.15,self.Tm) + self.H298
           self.H_fusion = H_Tm_solved_liquid - H_Tm_solved_solid

    def int_Cp_s_low(self,Tdown,Tup):
        Tdown,Tup = np.array([Tdown]),np.array([Tup])
        return (-2*self.A*Tup**3 - self.B*Tup**2 - self.C*Tup + \
        2*self.D*Tup**(-1)) - (-2*self.A*Tdown**3 - self.B*Tdown**2 - \
        self.C*Tdown + 2*self.D*Tdown**(-1))

    def int_Cp_s_hi(self,Tdown,Tup):
        Tdown,Tup = np.array([Tdown]),np.array([Tup])
        return (-self.E*Tup + 10*(self.F)*(Tup**(-9))) - (-self.E*Tdown +\
         10*(self.F)*(Tdown**(-9)))

    def int_Cp_l_low(self,Tdown,Tup):
        #Tdown,Tup = np.array([Tdown]),np.array([Tup])
        #print (Tdown,Tup,self.G,self.A,self.B,self.C,self.D)
        return (-6*self.G*Tup**7 - 2*self.A*Tup**3 - self.B*Tup**2 - \
        self.C*Tup + 2*self.D*Tup**-1) - \
        (-6*self.G*Tdown**7 - 2*self.A*Tdown**3 - self.B*Tdown**2 - self.C*Tdown + \
        2*self.D*Tdown**-1)

    def int_Cp_l_hi(self,Tdown,Tup):
        Tdown,Tup = np.array([Tdown]),np.array([Tup])
        return (-self.E*Tup) - (-self.E*Tdown)

    def H_solid(self,T):
        #print (np.array([self.int_Cp_s_low(298.15,Ti)[0] for Ti in T]))
        return np.array([self.int_Cp_s_low(298.15,Ti)[0] for Ti in T])

    def H_liquid(self,T):
        H_298_Tb_liquid = self.int_Cp_l_low(298.15,self.Tb)
        H_Tb_Ti_liquid = [self.int_Cp_l_hi(self.Tb,Ti)[0] for Ti in T]
        #print (H_298_Tb_liquid,self.H298,H_Tb_Ti_liquid)
        H_liquid_298_solid = np.array([H_298_Tb_liquid + self.H298 + h for h in \
        H_Tb_Ti_liquid])
        return H_liquid_298_solid

    def check_constraints(self):
        T = np.linspace(298.15,3000)
        for t in T:
            if self.Cp_solid([t])[0]<=0:
                _log.trace("failed +ve Cp")
                return False
            elif self.dCp_solid([t])[0]<0:
                _log.trace("failed +ve Cp slope")
                return False
            elif self.S_low_s(t)<0:
                _log.trace("negative entropy")
                return False
            elif self.S_hi_s(t)<0:
                _log.trace("negative entropy")
                return False
            elif self.S_low_l(t)<0:
                _log.trace("negative entropy")
                return False
            elif self.S_hi_l(t)<0:
                _log.trace("negative entropy")
                return False
        return True

    def evaluate_last(self):
        t = 500.0
        self.a = \
        self.G_low_s_integrated([t],298.15)[0] - (self.b*t + self.A*t**(3) + \
        self.B*t**(2) + self.C*t*np.log(t) + self.D*t**(-1))

        t = self.Tb + 100.0
        self.c = \
        self.G_hi_s_integrated([t],self.Tb)[0] - (self.d*t + self.E*t*np.log(t)+\
         self.F*t**-9)

        t = 500.0
        self.e = \
        self.G_low_l_integrated([t],298.15)[0] - (self.A*t**(3) + self.B*t**(2) + \
        self.C*t*np.log(t) + self.D*t**(-1) + self.G*t**7 + self.f*t - self.a)

        t = self.Tb + 100.0
        self.g = \
        self.G_hi_l_integrated([t],self.Tb)[0] - (self.h*t + self.E*t*np.log(t))

        self.f1 = self.f - self.b

    def dCp_solid(self,T):
        T_low = np.array([t for t in T if t <= self.Tb])
        T_hi = np.array([t for t in T if t > self.Tb])
        return list(-12*self.A*T_low -2*self.B*T_low**0 +6*self.D*T_low**(-3)) + \
               list(900*self.F*T_hi**(-11))


# BuILDS DATABASE FROM JSON FILES
#def get_unary_data(unary_path):
#    return glob(unary_path+"/*.json")

def load_unary_datasets(dataset_filenames) -> PickleableTinyDB:
    """
    Create a PickelableTinyDB with the data from a list of filenames.
    Parameters
    ----------
    dataset_filenames : [str]
        List of filenames to load as datasets
    Returns
    -------
    PickleableTinyDB
    """
    ds_database = PickleableTinyDB(storage=MemoryStorage)
    for fname in dataset_filenames:
        with open(fname) as file_:
            try:
                d = json.load(file_)
                # no particular checks required as of now
                ds_database.insert(clean_dataset(d))
            except ValueError as e:
                raise ValueError('JSON Error in {}: {}'.format(fname, e))
            except DatasetError as e:
                raise DatasetError('Dataset Error in {}: {}'.format(fname, e))
    return ds_database

if __name__ == "__main__":
    tdb_file = ""
    unary_path = ""
    datasets = glob(unary_path+"/*.json")
    unary_db = load_unary_datasets(datasets)
    unary_residual = SGTE91UnaryResidual(tdb_file,unary_db)

## Not sure if this part needs to be in this inheriting class in the module
#class ResidualRegistry():
#    def __init__(self) -> None:
#        self._registered_residual_functions: Type[ResidualFunction] = []

#    def get_registered_residual_functions(self) -> List[Type[ResidualFunction]]:
#        return self._registered_residual_functions

#    def register(self, residual_function: Type[ResidualFunction]):
        # Don't allow duplicates
#        if residual_function not in self._registered_residual_functions:
#            self._registered_residual_functions.append(residual_function)

#residual_function_registry = ResidualRegistry()
