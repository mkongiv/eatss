import sys
import re
import os
import math
import signal
import time

MEM_L1=(128*1024)
MEM_L2=(40*2**20)
MEM_SM=(64*1024)

MEM_STRIDE1=1
MEM_STRIDEX=0

EATSS_DIM_ALL=0
EATSS_DIM_ONLY_PAR=1
EATSS_OP_MUL=1
EATSS_OP_ADD=2
EATSS_ALIGNMENT_FACTOR=32
EATSS_ALIGNMENT_FRAC=1.0
EATSS_WARP_SIZE=16
EATSS_SHMEM_FRAC=1.0
GLOBAL_SOL_VAR='W_prog'

EATSS_MAX_TRIES=3
MEM2COMP_RATIO = 40 
EATSS_VERBOSE=True

PER_DIM = 1
PER_PROC = 2
USE_MODULO = False
DIM_UNMAPPED = -1
EATSS_DT = 'double'
BASE_INDENT='  '
DIM_NOT_USED=-2




def prod (ll):
  ret = 1
  for xx in ll:
    ret = ret * xx
  return ret

def iceil(num,den):
  return int(math.ceil(num/(1.0*den)))


class GPU:
  def __init__ (self, configfile):
    # T_P_W
    # W_P_S
    # T_P_S
    # B_P_S
    # R_P_S
    # R_P_B
    # R_P_T
    # T_P_B
    # cores
    self.resources = {}
    self.read_gpu_config (configfile)

  def read_gpu_config (self, configfile):
    ff = open (configfile)
    for ii,ll in enumerate(ff.readlines ()):
      ll = ll.strip ()
      part = ll.split(':')
      factor = 1
      if (ii >= 9):
        factor = 4
      dt_factor = 1
      if (EATSS_DT == 'double'):
        dt_factor = 2
      self.resources[part[0]] = int(part[1]) / (factor * dt_factor)
    self.resources['SH'] = int(self.resources['L1'] * EATSS_SHMEM_FRAC)
    print ('SHMEM % = {}'.format (EATSS_SHMEM_FRAC))
    for rr in self.resources:
      val = self.resources[rr]
      print ("Resource {} : {}".format (rr, val))
    print ('EAF at read_gpu_config: {}'.format (EATSS_ALIGNMENT_FRAC))
    #EATSS_ALIGNMENT_FACTOR = int(EATSS_ALIGNMENT_FRAC * self.resources['T_P_W'])
    if (self.resources['SH'] > self.resources['SHM_P_B']):
      print ("Illegal amount of shared-memory requested. Exceeded shared-memory per block ({}KB)".format (self.resources['SHM_P_B']))
      sys.exit (42)
    ff.close ()

  def show_gpu_config (self):
    print ("Legend: ")
    print ("S: SM")
    print ("B: thread Block")
    print ("T: Thread")
    print ("R: Register")
    print ("W: Warp")
    print ("Cores")
    for rr in sorted(self.resources):
      print ("{} : {}".format (rr, self.resources[rr]))

  def rec (self, met):
    if (met in self.resources):
      return self.resources[met]
    return '-1'

## Z3 optimization flags and optimization options.
class Comm_Opt_Form:
  def __init__ (self, output_filename):
    self.decl = []
    self.cstr = []
    self.modelfile = output_filename
    self.pvec = None
    self.options = ""
    self.options += "':algebraic_number_evaluator', False, "
    self.options += "':arith_ineq_lhs', False, " 
    #self.options += "':elim_to_real', True, "
    self.options += "':eq2ineq', False, " 
    self.options += "':expand_nested_stores', True, "
    self.options += "':gcd_rounding', False, "
    self.options += "':ignore_patterns_on_ground_qbody', True, "
    self.options += "':flat', False, "
    self.options += "':ite_extra_rules', True, "
    #self.options += "':max_memory', 7516192768, "
    #self.options += "':max_memory', 10737418240, "
    self.options += "':pull_cheap_ite', True, "
    self.options += "':push_ite_arith', True, "
    #self.options += "':push_to_real', False, "
    self.options += "':som', True, "
    self.options += "':som_blowup', 1000, "
    self.options += "':sort_store', True, "
    self.options += "':sort_sums', True, "
    self.options += "':split_concat_eq', True, "
    self.options += "':blast_select_store', True, "
    self.options += "':expand_select_ite', True"
    # testing
    #self.options += ",':local_ctx', True"
    #self.options += ",':cache_all', True"
    #self.options += ",':gcd_rounding', True"
    #self.options += ",':rewrite_patterns', True"
    #self.options += ",':expand_store_eq', True"
    self.options += ",':hoist_mul', True"
    self.options += ",':hoist_ite', True"
    #for dd,pp in enumerate(self.pvec):
    #  cstr = 'p{} == {}'.format (dd, pp)
    #  self.add_cstr (cstr)
    #  print ("Processor constraint : {}".format (cstr))

  def assemble_decl (self):
    ret = ""
    for dd in self.decl:
      if (not ret == ""):
        ret += "\n"
      ret = ret + dd
    return ret

  def assemble_cstr (self):
    ret = ""
    for cc in self.cstr:
      if (not ret == ""):
        ret += ", "
      ret = ret + cc
    return ret

  def print_decl_debug (self):
    variables = self.assemble_decl ()
    print ('Declared variables: {}'.format (variables))

  def print_cstr_debug (self):
    constraints = self.assemble_cstr ()
    print ('Formulation : {}'.format (constraints))

  def add_cstr (self, new_cstr):
    self.cstr.append (new_cstr)

  def add_cstr_shadow (self, mf, new_cstr):
    self.cstr.append (new_cstr)
    mf.write (new_cstr + '\n')

  def add_var (self, new_decl):
    self.decl.append (new_decl)

  def write_chunk (self, ff, chunk, chunk_id):
    cmnt = '## Chunk No. {} \n'.format (chunk_id)
    ff.write (cmnt)
    #ff.write (chunk)
    cmd = 'term = simplify (And ({}), {})\n'.format (chunk, self.options)
    #cmd = 'term = simplify (And ({}))\n'.format (chunk)
    ff.write (cmd)
    cmd = 'opt.add (term)\n'
    ff.write (cmd)
    ff.write ('\n')

  ## Write the COF to a python file script.
  def write_formulation (self, glob_obj_ub, n_fails):
    MAX_CHUNK = 250
    #MAX_CHUNK = 150
    variables = self.assemble_decl ()
    constraints = self.assemble_cstr ()
    ff = open (self.modelfile, 'w')
    ff.write ('from z3 import *\n')
    #ff.write ('opt = Optimize ()\n')
    ## qfnra = Quantifier Free Polynomial Real Arithmetic (e..g x^2 + y^2 < 1)
    ## qfnia = Quantifier Free Non-Linear Integer Arithmetic 
    ## ufnia = What does UF stand for? NIA = Non-Linear Integer Arithmetic 
    #ff.write ("opt = Then('simplify',With('ufnia',':arith.min',True),'qfnia','qfnra').solver ()\n")
    topts = ''
    topts += "':arith.min',True"
    #topts += ","
    #topts += "':arith.solver',2"
    topts += ","
    topts += "':arith.nl.rounds',1048576"
    topts += ","
    topts += "':arith.nl.delay',1000"
    topts += ","
    topts += "':qi.quick_checker',2"
    topts += ","
    topts += "':arith.nl.gr_q',50"
#
    topts += ","
    topts += "':algebraic_number_evaluator', False, "
    topts += "':arith_ineq_lhs', False, " 
    #topts += "':elim_to_real', True, "
    topts += "':eq2ineq', False, " 
    topts += "':expand_nested_stores', True, "
    topts += "':gcd_rounding', False, "
    topts += "':ignore_patterns_on_ground_qbody', True, "
    topts += "':flat', False, "
    topts += "':ite_extra_rules', True, "
    #topts += "':max_memory', 7516192768, "
    #topts += "':max_memory', 10737418240, "
    topts += "':pull_cheap_ite', True, "
    topts += "':push_ite_arith', True, "
    #topts += "':push_to_real', False, "
    topts += "':som', True, "
    topts += "':som_blowup', 1000, "
    topts += "':sort_store', True, "
    topts += "':sort_sums', True, "
    topts += "':split_concat_eq', True, "
    topts += "':blast_select_store', True, "
    topts += "':expand_select_ite', True"
    # testing
    #self.options += ",':local_ctx', True"
    #self.options += ",':cache_all', True"
    #self.options += ",':gcd_rounding', True"
    #self.options += ",':rewrite_patterns', True"
    #self.options += ",':expand_store_eq', True"
    topts += ",':hoist_mul', True"
    topts += ",':hoist_ite', True"
    #ff.write ("opt = Then('simplify',With('ufnia',':arith.min',True)).solver ()\n")
    #ff.write ("opt = Then(With('qfnra',arith.min=True)).solver ()\n")
    #ff.write ("opt = Then('simplify','ufnia').solver ()\n")
    ff.write ("opt = Then('simplify',With('ufnia',{})).solver ()\n".format (topts))
    #ff.write ('set_option (rational_to_decimal=True)\n')
    ff.write ('\n')
    ff.write (variables)
    ff.write ('\n')
    ff.write ('## Formulation Objectives\n')
    #ff.write ('K_obj = opt.minimize (K_prog)\n')
    #ff.write ('P_obj = opt.maximize (O_par)\n')
    #ff.write ('G_obj = opt.minimize (G_prog)\n')
    if (glob_obj_ub > 0):
      #nfails = 1
      base_scale = n_fails
      #left_scale = 1#base_scale + 1
      #right_scale = 1#base_scale + 2
      left_scale = base_scale + 1
      right_scale = base_scale + 2
      iter_g_obj_cstr = '{} * {} > {} * {}'.format (left_scale, GLOBAL_SOL_VAR, right_scale, glob_obj_ub)
      ff.write ('opt.add ({})\n'.format (iter_g_obj_cstr))
    ff.write ('\n')
    chunk = ""
    count = 0
    chunk_id = 1
    cache = {}
    for cc in self.cstr:
      if (cc in cache):
        #print ("[INFO] Skipping duplicated constraint: {}".format (cc))
        continue
      cache[cc] = 1
      if (count > 0):
        chunk += ", "
      count += 1
      count += cc.count (',') 
      chunk = chunk + cc
      if (count >= MAX_CHUNK):
        self.write_chunk (ff, chunk, chunk_id)
        count = 0
        chunk = ""
        chunk_id += 1
    # Write last chunk
    self.write_chunk (ff, chunk, chunk_id)   
    # Script epilogue
    #ff.write ('for k, v in opt.statistics ():\n')
    #ff.write ('  print ("K is {} - V is {}".format (k,v))\n')
    #ff.write ('\n')
    #ff.write ('print(opt.check())\n')
    ff.write ('if (opt.check() != unsat):\n')
    ff.write ('  sol = opt.model ()\n')
    ff.write ('  for vv in sol:\n')
    ff.write ('    print(vv, sol[vv])\n')
    ff.write ('else:\n')
    ff.write ('  print("unsat")\n')
    ff.close ()


class Reference:
  def __init__(self, form, gpu):
    self.name = ""
    self.np = None
    self.PP = None
    self.ndim = 0
    self.cof = form
    self.dims = {}
    self.sizes = {}
    self.map = {}
    self.data = None
    self.ref_id = 0
    self.gpu = gpu

  ## Read reference information from file
  def init_from_file (self, ff):
    line = ff.readline ()
    line = line.strip ()
    parts = line.split (':')
    self.name = parts[0]
    dimlist = parts[1].split (',')
    for dd,dname in enumerate(dimlist):
      self.dims[dd] = dname
      self.map[dd] = DIM_UNMAPPED
      self.ndim += 1
    sizes = parts[2].split(',')
    for dd,dsize in enumerate(sizes):
      self.sizes[dd] = dsize

  def get_dims (self):
    return self.dims

  def dimensionality (self):
    return len(self.dims)

  
  ## Show the array name and its sizes by printing it to stdout.
  def show_info(self):
    print ("  Reference: {} (ndim={})".format (self.name, self.ndim))
    for dd in self.dims:
      print ("  --> Dim {}: {} ({})".format (dd, self.dims[dd], self.sizes[dd]))


  def get_ref_dim (self):
    return len(self.dims)


  ## Method to add a generic pre-assembled constraint to the COF object
  ## and to the formulation file.
  def add_constraint (self, mf, cstr):
    self.writeln (mf, 'opt.add ({})'.format (cstr))
    self.cof.add_cstr (cstr)

  ## Return the name of the array
  def get_name (self):
    return self.name

  ## Return the number of dimensions of the current array object.
  def get_num_dim (self):
    return len(self.dims)

  def get_iter_name (self, adim):
    return self.dims[adim]

    
  def writeln(self, mf, line):
    mf.write (line + "\n")

  def set_lower_bound (self, mf, varname, lb):
    cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_upper_bound (self, mf, varname, ub):
    cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds (self, mf, varname, lb, ub):
    cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    #cmd = 'opt.add ({} >= {}, {} <= {})'.format (varname, lb, varname, ub)
    #self.writeln (mf, cmd)
    self.set_bounds (mf, varname, lb, ub)


  def declare_variable (self, mf, varname, decl):
    if (decl == None):
      print ("Exiting")
      sys.exit(42)
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname  
    return decl


  # Return variable for computing data slices.
  def get_block_size_var (self, dim):
    varname = 'DS_{}_{}'.format (self.name, dim)
    return varname

  def get_rho_varname (self):
    return 'rho_{}'.format (self.name)

  def get_rho_dim_varname (self, adim):
    return 'rho_{}_i{}'.format (self.name, adim)


  def bound_replication_variables (self, mf):
    rho_var = self.get_rho_varname ()
    self.set_bounds_boolean (mf, rho_var)
    for dd in self.dims:
      rho_var = self.get_rho_dim_varname (dd)
      self.set_bounds_boolean (mf, rho_var)



  ## Return the array dimension size, as an integer, given the associated
  ## iterator name used to access it in the rels input file.
  def get_array_extent_by_dim_name (self, dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == dim_name):
        return int(self.sizes[dd])
    return -1

  def set_block_function (self, mf, bvar, dim, pbs):
    pass

  def declare_block_variables (self, mf, decl):
    for dd in range(self.ndim):
      varname = self.get_block_size_var (dd)
      decl = self.declare_variable (mf, varname, decl)
      size = self.sizes[dd]
      #print ("Ref {} {} = {}".format (self.name, dd, self.sizes[dd]))
      self.set_upper_bound (mf, varname, size)
      self.set_block_function (mf, varname, dd, size)
    return decl

  def is_dim_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return True
    return False

  def get_dim_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return dd
    return -1

  def get_dim_size_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return int(self.sizes[dd])
    return 0


  ## Return the name of the capacity constraint for a given statement.
  def get_volume_var (self):
    varname = 'req_{}'.format (self.name)
    return varname



  def get_tile_vol (self, stmt):
    return self.get_local_volume (stmt)


  def get_array_vol_var (self, stmt_name):
    varname = 'vol_{}_{}'.format (stmt_name, self.name)
    return varname

  def declare_array_vol_var (self, mf, decl, stmt_name):
    varname = self.get_array_vol_var (stmt_name)
    return self.declare_variable (mf, varname, decl)

  def get_tile_dim_var (self, adim):
    varname = 'T_{}'.format (adim)
    return varname

  def define_array_vol_constraints (self, mf, decl, stmt_name):
    vol_var = self.get_array_vol_var (stmt_name)
    decl = self.declare_array_vol_var (mf, decl, stmt_name)
    cstr = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (cstr != ''):
        cstr += ' * '
      tile_dim_var = self.get_tile_dim_var (iter_name)
      decl = self.declare_variable (mf, tile_dim_var, decl)
      bound = '{} >= 1'.format (tile_dim_var)
      self.add_constraint (mf, bound)
      bound = '{} <= {}'.format (tile_dim_var, self.gpu.rec('T_P_B'))
      self.add_constraint (mf, bound)
      cstr += tile_dim_var
      bound = '{} >= {}'.format (vol_var, tile_dim_var)
      self.add_constraint (mf, bound)
      bound = '{} <= 999999'.format (vol_var)
      bound = '{} % {} == 0'.format (tile_dim_var, EATSS_ALIGNMENT_FACTOR)
      self.add_constraint (mf, bound)
    cstr = '{} == {}'.format (vol_var, cstr)
    self.add_constraint (mf, cstr)
    return decl

  def is_iterator_stride_1 (self, dim_name):
    ndim = len(self.dims)
    for dd in self.dims:
      adim = self.dims[dd]
      if (ndim >= 2 and adim == dim_name and dd == ndim - 1):
        return 1
    return 0


## Start of Statement Class
class Statement:
  def __init__(self, form, gpu):
    self.name = ""
    self.PP = None
    self.cof = form
    self.ndim = 0
    self.dims = {}
    self.par = {}
    self.nref = 0
    self.refs = {}
    self.accs = [] # same as refs but a list
    self.map = {}
    self.gpu = gpu

  ## Read statement information from file
  def init_from_file (self, ff):
    line = ff.readline ()
    line = line.strip ()
    parts = line.split (':')
    self.name = parts[0]
    dimlist = parts[1].split (',')
    for dd,dname in enumerate(dimlist):
      self.par[dd] = 0
      if (dname.find('*')>0):
        self.par[dd]=1
      dname = re.sub ('\*','',dname)
      self.dims[dd] = dname
      self.map[dd] = DIM_UNMAPPED
      self.ndim += 1
    line = ff.readline ()
    line = line.strip ()
    self.nref = int(line)
    for aa in range(self.nref):
      ref = Reference (self.cof, self.gpu)
      ref.init_from_file (ff)
      self.refs[ref.get_name ()] = ref
      self.accs.append (ref)
    
  #def collect_arrays (self, collection):
  #  for aa in self.accs:
  #    print (aa)
  #    collection[aa.get_name ()] = aa

  ## Return the number of dimensions of the current statement object.
  def get_num_dim (self):
    return len(self.dims)
    
  def get_ref (self, refid):
    if (refid >= len(self.accs)):
      return None
    return self.accs[refid]

  def get_ref_by_name (self, ref_name):
    for ref in self.accs:
      if (ref.get_name () == ref_name):
        return ref
    return None

  def show_info(self):
    print ("Statement: {} (ndim={})".format (self.name, self.ndim))
    for dd in self.dims:
      partag=''
      if (self.par[dd] == 1):
        partag='*'
      print ("Dim {}: {}{}".format (dd, self.dims[dd],partag))
    for aa in self.refs:
      self.refs[aa].show_info ()

  def get_dims (self):
    return self.dims


  # Find the loop dimension corresponding to an iterator
  def get_dim_by_name (self, iter_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_name):
        return dd
    return -1


  def get_name (self):
    return self.name

  ## Return the loop tripcount associated to the given dimension id.
  ## The argument dim_id must be between 0 and (depth-1).
  def get_loop_dim_tripcount (self, dim_id):
    dim_name = self.dims[dim_id]
    for ref in self.accs:
      if (ref.is_dim_used (dim_name)):
        return ref.get_dim_size_if_used (dim_name)
    return 0


  def writeln(self, mf, line):
    mf.write(line + "\n")

  def add_constraint (self, mf, cstr):
    self.writeln (mf, 'opt.add ({})'.format (cstr))
    self.cof.add_cstr (cstr)

  def set_lower_bound (self, mf, varname, lb):
    plain_cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_upper_bound (self, mf, varname, ub):
    plain_cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_bounds (self, mf, varname, lb, ub):
    plain_cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    #plain_cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    #cmd = 'opt.add ({})'.format (plain_cstr)
    #self.writeln (mf, cmd)
    #self.cof.add_cstr (plain_cstr)
    self.set_bounds (mf, varname, lb, ub)

  def declare_variable (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_boolean (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Bool('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  ## 
  ## Energy formulation per statement
  ##

  def get_loop_dim (self):
    return len(self.dims)

  def get_tile_dim_var (self, adim):
    varname = 'T_{}'.format (adim)
    return varname

  def get_stmt_vol_var (self):
    varname = 'vol_{}'.format (self.name)
    return varname

  def declare_vol_var (self, mf, decl):
    varname = self.get_stmt_vol_var ()
    return self.declare_variable (mf, varname, decl)

  def set_vol_var_bounds (self, mf):
    vol_var = self.get_stmt_vol_var ()

  def set_stmt_vol_expr (self, mf, decl):
    decl = self.declare_vol_var (mf, decl)
    vol_var = self.get_stmt_vol_var ()
    cstr = ''
    print ("At stmt {}".format (self.name))
    for aa in self.refs:
      print ("At array {}".format (aa))
      ref = self.refs[aa]
      decl = ref.define_array_vol_constraints (mf, decl, self.name)
      print ("\t Define vol constraints...")
      if (cstr != ''):
        cstr += ' + '
      ref_vol = ref.get_array_vol_var (self.name)
      cstr += ref_vol
      bound = '{} >= {}'.format (vol_var, ref_vol)
      print ("\t Setting bounds of {}...".format (vol_var))
      self.add_constraint (mf, bound)
    cstr = '{} == {}'.format (vol_var, cstr)
    self.add_constraint (mf, cstr)
    return decl
    
  def find_stride_1_loop (self):
    res = -1
    best_score = 0
    for dd in self.dims:
      dim_name = self.dims[dd]
      score = 0
      for aa in self.refs:
        ref = self.refs[aa]
        score += ref.is_iterator_stride_1 (dim_name)
      print ("Array {} : {}".format (aa, score))
      if (score > best_score):
        best_score = score
        res = dd
    try:
      print ("Best loop: {}, level={} ({})".format (self.dims[res], res, best_score))
    except KeyError:
      sys.exit(1)
    return res

  def count_parallel_loops (self):
    ret = 0
    for dd in self.dims:
      ret += self.par[dd]
    return ret

  def count_parallel_stride1_loops (self):
    ret = 0
    for dd in self.dims:
      dim_name = self.dims[dd]
      score = 0
      for aa in self.refs:
        ref = self.refs[aa]
        score += ref.is_iterator_stride_1 (dim_name) * (ref.dimensionality () - 1)
      if (score > 1):
        score = 1
      ret += self.par[dd] * score
    return ret

  ## @Statement
  def collect_stride_weights (self):
    ret = []
    s1loop = self.find_stride_1_loop ()
    nps1l = self.count_parallel_stride1_loops ()
    choice = 0
    if (nps1l < 1):
      choice = 1
    for dd in self.dims:
      dim_name = self.dims[dd]
      score = 0
      for aa in self.refs:
        ref = self.refs[aa]
        score += ref.is_iterator_stride_1 (dim_name)
      factor = 1
      if (dd == s1loop):
        factor = EATSS_ALIGNMENT_FACTOR
      weight = (score - self.par[dd]) * score * factor
      print ('Iterator {} : par={}, nps1l={}, score={}, factor={}'.format (dim_name, self.par[dd], nps1l, score, factor))
      #if (weight == 0):
      #  weight = 1
      ret.append (weight)
    return ret

  def build_weighted_tile_size_sum_expr (self):
    cstr = ''
    weights = self.collect_stride_weights ()
    for dd in self.dims:
      dim_name = self.dims[dd]
      if (cstr != ''):
        cstr += ' + '
      tsvar = self.get_tile_dim_var (dim_name)
      cstr += '({} * {})'.format (weights[dd], tsvar)
    return cstr
    


  def assign_mem_type_to_arrays (self):
    s1loop = self.find_stride_1_loop ()
    s1iter = self.dims[s1loop]
    for aa in self.refs:
      ref = self.refs[aa]
      is_s1 = ref.is_iterator_stride_1 (s1iter)
      mem_type = ""
      if (is_s1 == 1):
        mem_type = 'cache'
      else:
        mem_type = 'shared-memory'
      print ("Array {} : {}".format (aa, mem_type))

  def get_stmt_memory_type_vol_var (self, prefix):
    varname = "{}_{}".format (prefix, self.name)
    return varname

  def declare_stmt_memory_type_vol_var (self, mf, decl, prefix):
    varname = self.get_stmt_memory_type_vol_var (prefix)
    decl = self.declare_variable (mf, varname, decl)
    return decl

    
  def set_memory_type_constraint (self, mf, decl, mem_type, prefix, fraction_used):
    s1loop = self.find_stride_1_loop ()
    s1iter = self.dims[s1loop]
    decl = self.declare_stmt_memory_type_vol_var (mf, decl, prefix)
    stmt_vol = self.get_stmt_memory_type_vol_var (prefix)
    cstr = ""
    for aa in self.refs:
      ref = self.refs[aa]
      is_s1 = ref.is_iterator_stride_1 (s1iter)
      #if (is_s1 == mem_type or fraction_used == 1.0):
      if (is_s1 == mem_type):
        if (cstr != ""):
          cstr += " + "
        ref_vol = ref.get_array_vol_var (self.name)
        bound = '{} >= {}'.format (stmt_vol, ref_vol)
        self.add_constraint (mf, bound)
        cstr += ref_vol
    if (cstr != ''):
      cstr = '{} == {}'.format (stmt_vol, cstr)
      self.add_constraint (mf, cstr)
    return decl

  def set_shared_memory_constraint (self, mf, decl):
    fraction = EATSS_SHMEM_FRAC
    return self.set_memory_type_constraint (mf, decl, MEM_STRIDEX, "shared", fraction)

  def set_l2_memory_constraint (self, mf, decl):
    fraction = EATSS_SHMEM_FRAC
    return self.set_memory_type_constraint (mf, decl, MEM_STRIDE1, "l2", 1.0 - fraction)

  def set_l1_memory_constraint (self, mf, decl):
    fraction = EATSS_SHMEM_FRAC
    return self.set_memory_type_constraint (mf, decl, MEM_STRIDE1, "l1", 1.0 - fraction)

  def get_block_size_var (self):
    varname = 'stmt_{}_blk_size'.format (self.name)
    return varname

  def declare_block_size_var (self, mf, decl):
    limvar = self.get_block_size_var ()
    decl = self.declare_variable (mf, limvar, decl)
    return decl

  # Return the product of tile sizes
  def get_tile_size_product (self, only_par=True):
    ret = 1
    for dd in self.dims:
      ff = self.dims[dd]
      if ((only_par and self.par[dd] == 1) or (not only_par)):
        ret = ret * ff
    return ret

  def get_tile_size_operator_expr (self, filter_dim, operator):
    cstr = ''
    opstr = ' * '
    if (operator == EATSS_OP_ADD):
      opstr = ' + '
    for dd in self.dims:
      idim = self.dims[dd]
      tsvar = self.get_tile_dim_var (idim)
      if (filter_dim == EATSS_DIM_ONLY_PAR and self.par[dd] == 1):
        if (cstr != ''):
          cstr += opstr
        cstr += tsvar
      elif (filter_dim == EATSS_DIM_ALL):
        if (cstr != ''):
          cstr += opstr
        cstr += tsvar
    return cstr


  def get_tile_size_product_expr (self, filter_dim):
    return self.get_tile_size_operator_expr (filter_dim, EATSS_OP_MUL)

  def get_tile_size_sum_expr (self, filter_dim):
    return self.get_tile_size_operator_expr (filter_dim, EATSS_OP_ADD)
    

  # Set the expression determining the number of threads per thread block
  def set_block_size_expr (self, mf, decl):
    decl = self.declare_block_size_var (mf, decl)
    blk_size_var = self.get_block_size_var ()   
    only_par = True
    cstr = self.get_tile_size_product_expr (EATSS_DIM_ONLY_PAR)
    cstr = '{} == {}'.format (blk_size_var, cstr)
    self.add_constraint (mf, cstr)
    return decl

  def get_intra_thread_work_var (self):
    varname = 'W_{}'.format (self.name)
    return varname

  def declare_intra_sm_work_var (self, mf, decl):
    varname = self.get_intra_thread_work_var ()
    decl = self.declare_variable (mf, varname, decl)
    return decl

  def maximize_work_per_sm (self, mf, decl):
    decl = self.declare_intra_sm_work_var (mf, decl)
    w_var = self.get_intra_thread_work_var ()
    cstr = self.get_tile_size_product_expr (EATSS_DIM_ONLY_PAR)
    for dd in self.dims:
      idim = self.dims[dd]
      tsvar = self.get_tile_dim_var (idim)
      bound = '{} >= {}'.format (w_var, tsvar)
      self.add_constraint (mf, bound)
    sum_weight = self.build_weighted_tile_size_sum_expr ()
    #cstr = '{} == {}'.format (w_var, cstr)
    cstr = '{} == {} + {}'.format (w_var, cstr, sum_weight)
    self.add_constraint (mf, cstr)
    return decl

    

  ## Set several resource limits per statement
  def set_resource_limits (self, mf, decl):
    # Set limit on shared memory
    l1_remaining = self.gpu.rec('L1') - self.gpu.rec('SH')
    if (self.gpu.rec('SH') > 0):
      vol_var = self.get_stmt_memory_type_vol_var ('shared')
      cstr = '{} <= {}'.format (vol_var, self.gpu.rec('SH'))
      self.add_constraint (mf, cstr)
      cstr = '{} <= {}'.format (vol_var, self.gpu.rec('SHM_P_B'))
      self.add_constraint (mf, cstr)
    # Set limit on l2 cache only if L1 is used
    #if (l1_remaining > 0):
    vol_var = self.get_stmt_memory_type_vol_var ('l2')
    cstr = '{} <= {}'.format (vol_var, self.gpu.rec('L2'))
    self.add_constraint (mf, cstr)
    # Set limit on L1 cache
    if (l1_remaining > 0):
      vol_var = self.get_stmt_memory_type_vol_var ('l1')
      cstr = '{} <= {}'.format (vol_var, l1_remaining)
      self.add_constraint (mf, cstr)
    # Set limit of register per sm
    reg_lim = self.get_block_size_var ()
    factor = 1
    if (EATSS_DT == 'double'):
      factor = 2
    nrefs = len(self.refs)
    cstr = '{} <= {} / ({} * {})'.format (reg_lim, self.gpu.rec('R_P_S'), nrefs, factor)
    self.add_constraint (mf, cstr)


  ## Traverse the list of accesses and add the array names to the
  ## @arrset dictionary. Return the updated dictionary.
  def collect_arrays (self, arrset):
    for ref in self.accs:
      if (ref.get_name () in arrset):
        continue
      arrset[ref.get_name ()] = ref
    return arrset

  def get_iterator_variable (self, idim, is_point):
    varname = 'i{}'.format (idim)
    if (not is_point):
      varname = 't{}'.format (idim)
    return varname


  def generated_array_name (self):
    varname = re.sub ('^gen','',self.name)
    return varname




## Global routines used to write the script and read the solution.

def declare_variable (mf, varname, decl, cof):
  decl[varname] = varname
  decl_cmd = '{} = Int (\'{}\') \n'.format (varname, varname)
  mf.write (decl_cmd)
  cof.add_var (decl_cmd)
  return decl

def declare_float (mf, varname, decl, cof):
  decl[varname] = varname
  decl_cmd = '{} = Real (\'{}\') \n'.format (varname, varname)
  mf.write (decl_cmd)
  cof.add_var (decl_cmd)
  return decl


##def set_tile_size_global_constraints (SS, mf, decl, cof):
##  ldims = []
##  for ss in SS:
##    stmt = SS[ss]
##    dim = stmt.get_loop_dim ()
##    ldims.append (dim)
##  max_dim = max(ldims)
##  for ts in range(max_dim):

    

def set_global_intra_thread_work_objective (SS, mf, decl, cof):
  obj_var = 'W_prog'
  decl = declare_variable (mf, obj_var, decl, cof)
  cstr = ''
  for sid in SS:
    stmt = SS[sid]
    if (cstr != ''):
      cstr += ' + '
    w_var = stmt.get_intra_thread_work_var ()
    cstr += w_var
    bound = '{} >= {}'.format (obj_var, w_var)
    cof.add_cstr_shadow (mf, bound)
  cstr = '{} == {}'.format (obj_var, cstr)
  cof.add_cstr (cstr)
  z3_cmd = 'opt.add ({})\n'.format (cstr)
  mf.write (z3_cmd)
  return decl



def read_solution_from_file (solfile):
  ff = open (solfile)
  ret = {}
  for line in ff.readlines ():
    if (line[0] != '('):
      continue
    line = line.strip ()
    line = re.sub (" ","",line)
    line = re.sub ("\(","",line)
    line = re.sub ("\)","",line)
    if (EATSS_VERBOSE):
      print (line)
    if (line.find ("unsat") >= 0):
      ff.close ()
      return None
    if (line == "sat"):
      continue
    #if (line.find ("div") >= 0):
    #  continue
    #if (line.find ("mod") >= 0):
    #  continue
    if (line.find ("->") >= 0):
      continue
    parts = line.split (",")
    if (parts[0].find ('T_') == 0):
      ret[parts[0]] = parts[1] 
    if (parts[0].find ('W_') == 0):
      ret[parts[0]] = parts[1] 
  ff.close ()
  return ret


def show_solution_from_table (solset):
  for kk in sorted(solset):
    print ("{} : {}".format (kk, solset[kk]))

def store_solution_to_file (solset, solfile):
  ff = open (solfile, "w")
  for kk in sorted(solset):
    ff.write ("{}:{}\n".format (kk, solset[kk]))
  ff.close ()


def show_tile_sizes (solset, stmt, shmem_frac, infile):
  dims = stmt.get_dims ()
  bench = re.sub ('\.rels','',infile)
  ret = ''
  for dd in dims:
    dim_name = dims[dd]
    tilevar = stmt.get_tile_dim_var (dim_name)
    ts = solset[tilevar]
    if (ret != ''):
      ret += ' '
    ret += ts
  # pretty print benchmark name
  if '/' in bench:
    bench = bench.split('/')[-1]
  print('TILE_CONF:{}:{}:{}:{}'.format (bench,shmem_frac,EATSS_ALIGNMENT_FRAC,ret))


def collect_tile_sizes (solset, stmt, shmem_frac, infile):
  dims = stmt.get_dims ()
  bench = re.sub ('\.rels','',infile)
  ret = ''
  for dd in dims:
    dim_name = dims[dd]
    tilevar = stmt.get_tile_dim_var (dim_name)
    ts = solset[tilevar]
    if (ret != ''):
      ret += ','
    ret += ts
  return ret

def solve_problem (modelfile, solfile, it):
  print ("Using Z3-solver")
  print ("Model file: {}".format (modelfile))
  cmd = 'python {} | sort > {}'.format (modelfile, solfile)
  os.system (cmd)
  curr_sol = read_solution_from_file (solfile) 
  if (curr_sol != None):
    os.system ('cp {} {}.{}'.format (solfile, solfile, it))
  return curr_sol


##################################################################################
##
##        Main driver starts here.
##
##################################################################################

if (len(sys.argv) < 4 or len(sys.argv) > 5):
  print ("python input-file.rels gpu-config-file shm-fraction warp-fraction")
  print ("Example: python gemm.rels a100.gpu 1.0 0.5")
  sys.exit(42)

infile  = sys.argv[1]
gpu_config_file = sys.argv[2]
EATSS_SHMEM_FRAC = float(sys.argv[3])

if (len(sys.argv) == 5):
  EATSS_ALIGNMENT_FRAC = float(sys.argv[4])

gpu = GPU (gpu_config_file)
gpu.show_gpu_config ()
EATSS_ALIGNMENT_FACTOR = int(EATSS_ALIGNMENT_FRAC * gpu.rec('T_P_W'))

call_solver = True

print ("Infile : {}".format (infile))
ff = open (infile, "r")

modelfile = re.sub ('\.rels','.model.py', infile)
solfile = re.sub ('\.rels','.sol', infile)

mf = open (modelfile + '.shadow', "w")
mf.write('from z3 import *\n\n')
mf.write("opt = Then('simplify','ufnia','qfnra').solver ()\n\n")

#mf.write (cmd)

# Formulation object
form = Comm_Opt_Form (modelfile)

nstmt = int (ff.readline())
SS = {}
CG = [] # control graph
AA = {}
for ss in range(nstmt):
  stmt = Statement (form, gpu)
  stmt.init_from_file (ff)
  # Gather all the arrays in a separate collection.
  AA = stmt.collect_arrays (AA)
  SS[stmt.get_name()] = stmt
  CG.append (stmt)

## Show info for each statement.
for ss in SS:
  stmt = SS[ss]
  stmt.show_info ()
 
#for name in AA:
#  aa = AA[name]
#  print ('Array {} - :{}:'.format (name, aa.get_name ()))

ff.close ()

decl = {}

## Start energy formulation
mf.write ("\n## Set tile volume expressions\n")
for ss in SS:
  stmt = SS[ss]
  decl = stmt.set_stmt_vol_expr (mf, decl)

mf.write ("\n## Set expressions aggregating types of memory per statement\n")
for ss in SS:
  stmt = SS[ss]
  stmt.find_stride_1_loop ()
  stmt.assign_mem_type_to_arrays ()

if (gpu.resources['SH'] > 0):
  mf.write ('\n# Set expressions for shared-memory usage per statement\n')
  for ss in SS:
    stmt = SS[ss]
    decl = stmt.set_shared_memory_constraint (mf, decl)

l1_remaining = gpu.resources['L1'] - gpu.resources['SH']
if (l1_remaining > 0):
  mf.write ('\n# Set expressions for L1 memory usage per statement\n')
  for ss in SS:
    stmt = SS[ss]
    decl = stmt.set_l1_memory_constraint (mf, decl)

if (gpu.resources['L2'] > 0):
#if (l1_remaining > 0):
  mf.write ('\n# Set expressions for L2 memory usage per statement\n')
  for ss in SS:
    stmt = SS[ss]
    decl = stmt.set_l2_memory_constraint (mf, decl)


mf.write ('\n# Set expression for number of threads per thread block\n')
for ss in SS:
  stmt = SS[ss]
  decl = stmt.set_block_size_expr (mf, decl)

mf.write ('\n# Set expression for intra-thread work per statement\n')
for ss in SS:
  stmt = SS[ss]
  decl = stmt.maximize_work_per_sm (mf, decl)

mf.write ('\n# Set several resource limits\n')
for ss in SS:
  stmt = SS[ss]
  stmt.set_resource_limits (mf, decl)

set_global_intra_thread_work_objective (SS, mf, decl, form)

opt_val = 0
n_fails = 0
form.write_formulation (opt_val, n_fails)
if (decl == None):
  print ("decl is None at 4182")
  sys.exit (42)

mf.close ()


prog_name = ''
for prog_name in SS:
  pass

tstart=time.time ()

opt_val = 0
solset = None
it = 1
max_tries = EATSS_MAX_TRIES
n_fails = 0
g_sols = []
while (call_solver and n_fails < max_tries):
  form.write_formulation (opt_val, n_fails)
  curr_sol = None
  if (call_solver):
    #print ("Using Z3-solver")
    #print ("Model file: {}".format (modelfile))
    #cmd = 'time python {} | sort > {}'.format (modelfile, solfile)
    #os.system (cmd)
    ## NOTE: G_prog is has been temporarily removed.
    #cmd = 'grep G_prog {} > a.sol'.format (solfile)
    #os.system (cmd)
    ##sys.exit (42)
    #curr_sol = read_solution_from_file (solfile) 
    if (curr_sol != None):
      os.system ('rm -f {}'.format (solfile))
    curr_sol = solve_problem (modelfile, solfile, it)
    os.system ('cat ' + solfile)
    if (curr_sol != None):
      show_solution_from_table (curr_sol)
    if (curr_sol == None or not GLOBAL_SOL_VAR in curr_sol):
      n_fails += 1
      continue
      #break
    #print (curr_sol)
    if (curr_sol != None and GLOBAL_SOL_VAR in curr_sol):
      new_opt_val = int(curr_sol[GLOBAL_SOL_VAR])
      tile_list=collect_tile_sizes (curr_sol, SS[prog_name], EATSS_SHMEM_FRAC, infile)
      sys.stdout.flush ()
      sys.stderr.flush ()
      print ('SUMMARY:{}:{}:SHM={}:EAF={}:{}'.format (infile, it, EATSS_SHMEM_FRAC, EATSS_ALIGNMENT_FRAC, tile_list))
      sys.stdout.flush ()
      sys.stderr.flush ()
      if (opt_val == 0):
        solset = curr_sol
        opt_val = new_opt_val
        g_sols.append (opt_val)
        print ("Found first solution : {}".format (opt_val))
      elif (new_opt_val > opt_val):
        solset = curr_sol
        opt_val = new_opt_val
        g_sols.append (opt_val)
        #opt_val = read_solution ('a.sol')
        print ("Found improved solution : {}".format (opt_val))
      else:
        print ("No new solution found, reducing step to: {} {} vs {} {}".format (n_fails + 2, GLOBAL_SOL_VAR, n_fails + 1, opt_val))
        n_fails += 1
      print ("Iteration #{} - Solution found: {}".format (it, opt_val))
      it += 1
      print ("------------------------------------------------------------------")

tstop=time.time ()

name = ''
for name in SS:
  pass
print ("No. of attempted tries: {}".format (n_fails))
if (solset == None):
  print ("No solution found.")
  print ('EATSS_ALIGNMENT_FACTOR:{}'.format (EATSS_ALIGNMENT_FACTOR))
  print ('EATSS_ALIGNMENT_FRAC:{}'.format (EATSS_ALIGNMENT_FRAC))
  print ('args={}'.format(len(sys.argv)))
  sys.exit (1)
print ('Solution found for {} (SHM={}) : {}'.format (infile, EATSS_SHMEM_FRAC, solset))
num_dim=SS[name].get_num_dim ()
print ('Solving time (TST):{}:{}:{}:{}'.format (infile,num_dim,it-1,tstop-tstart))
print ('Average Solving Time (AST):{}:{}:{}'.format (infile,num_dim,(tstop-tstart)/(it-1)))
show_tile_sizes (solset, SS[name], EATSS_SHMEM_FRAC, infile)
#store_solution_to_file (solset, solfile)

for ii,gg in enumerate(g_sols):
  print ("Sol.{} : {}".format (ii, gg))

print ('\nWARP FRACTION: {}'.format (EATSS_ALIGNMENT_FRAC))
