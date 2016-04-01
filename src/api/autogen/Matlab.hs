module Matlab (generate, synopsis, ctorSynopsis) where

import Control.Monad (zipWithM, zipWithM_, when, liftM)
import Data.Char (toUpper, toLower, isUpper)
import Data.List (intercalate, intersperse)
import Data.Maybe (catMaybes)
import Data.String.Utils
import Text.Printf
import Text.PrettyPrint
import System.IO
import System.FilePath

import API
-- TODO: remove if possible
import Codegen
import qualified Common
import qualified C


{- The matlab API consists of:
 - 1. A MEX file which wraps calls to underlying C++ objects
 - 2. A number of m-files which calls the m-file and includes
 -    function documentation -}
generate :: [ApiModule]
    -> [ApiFunction] -- ^ Functions common to both network and simulation modules
    -> IO ()
generate modules constructableFunctions = do
    generateMex (zip moduleNames functionDefs) constructableFunctions
    zipWithM_ generateMatlabFile (functionDefs ++ constructableFunctions) [0..]
    generateCMake (functionDefs ++ constructableFunctions)
    generateHelp modules
    where
        -- all functions, regardless of module
        functionDefs = concatMap mdl_functions modules

        -- the corresponding module names
        moduleNames =
            let go m = replicate (length $ mdl_functions m) (mdl_name m)
            in concatMap go modules


{- | Create the top-level help file. -}
generateHelp :: [ApiModule] -> IO ()
generateHelp modules = do
    withFile "../matlab/nemo.m" WriteMode $ \h -> do
    hPutStr h =<< readFile "../matlab/sources/nemo.m.header"
    example <- readFile "../matlab/example.m"
    hPutStr h $ unlines $ map ("%    " ++) $ lines example
    hPutStr h =<< readFile "../matlab/sources/nemo.m.footer"
    hPutStr h $ render $ vcat $ intersperse (char '%') $ map moduleHelp modules
    hPutStr h "\n"
    where
        moduleHelp :: ApiModule -> Doc
        moduleHelp m = (commentLine $ text $ stage m) <> char ':' $+$ moduleFunctions m

        moduleFunctions :: ApiModule -> Doc
        moduleFunctions m = vcat $ map (commentLine . (char ' ' <>) . text . matlabFunctionName) $ allModuleFunctions m

        stage :: ApiModule -> String
        stage m = case name m of
            "Network" -> "Construction"
            _ -> name m


{- | Auto-generate the list of m-files that cmake should install. This includes
 - both auto-generated and hand-written code -}
generateCMake :: [ApiFunction] -> IO ()
generateCMake fs =
    withFile fileName WriteMode $ \h -> do
    hPutStr h $ intercalate "\n" $ map matlabFileName fs
    where
        fileName = ".." </> "matlab" </> "install" <.> "cmake"



{- | Write a complete MEX file containing all code -}
generateMex :: [(String, ApiFunction)] -> [ApiFunction] -> IO ()
generateMex functions constructableFunctions = do
    withFile "../matlab/nemo_mex.cpp" WriteMode $ \mex -> do
    insertFileContents mex "../matlab/sources/nemo.header.mex"
    hPutStr mex $ render $ vcat [
        C.comment $ text "AUTO-GENERATED CODE START",
        Common.emptyLine,
        (vcat $ map (uncurry mexFunction) functions),
        (vcat $ map mexConstructibleFunction constructableFunctions),
        (mexFunctionTable $ map mexFunctionName $ (map snd functions) ++ constructableFunctions),
        C.comment $ text "AUTO-GENERATED CODE END"
      ]
    insertFileContents mex "../matlab/sources/nemo.footer.mex"
    where
        insertFileContents :: Handle -> String -> IO ()
        insertFileContents h filename = hPutStr h =<< hGetContents =<< openFile filename ReadMode



{- Return C/MEX code for a single function -}
mexFunction
    :: String       -- ^ module name/type
    -> ApiFunction
    -> Doc
mexFunction mname fn =
    if (MEX `elem` fn_noauto fn)
        then empty
        else if fn_vectorized fn
            then mexVectorFunction mname fn
            else mexScalarFunction mname fn

{- Return C/MEX code for a single function which is found in both Network and Simulation class -}
mexConstructibleFunction :: ApiFunction -> Doc
mexConstructibleFunction fn =
    if (MEX `elem` fn_noauto fn)
        then empty
        else if fn_vectorized fn
            then mexVectorConstructibleFunction fn
            else mexScalarConstructibleFunction fn




{- | Generate an m-file for a single function -}
generateMatlabFile :: ApiFunction -> Int -> IO ()
generateMatlabFile fn fn_id = do
    code <- if Matlab `elem` fn_noauto fn
                then includeMatlabFunction (name fn) fn_id
                else return $! render $ matlabFunction fn fn_id
    withFile (".." </> "matlab" </> matlabFileName fn) WriteMode $ \h -> hPutStr h code


{- Some mfiles are hand-written due to special constraints on input or output
 - format. These are simply included with the function index substituted as
 - appropriate -}
includeMatlabFunction :: String -> Int -> IO String
includeMatlabFunction basename fnIdx = do
    fnDef <- readFile $ ".." </> "matlab" </> "sources" </> basename <.> "in.m"
    return $! replace "FNID" (show fnIdx) fnDef



matlabFunctionName :: ApiFunction -> String
matlabFunctionName fn = Common.camelCasedName ["nemo", name fn]


matlabFileName :: ApiFunction -> String
matlabFileName fn = matlabFunctionName fn <.> "m"


{- | Code and documentation for a single function -}
matlabFunction :: ApiFunction -> Int -> Doc
matlabFunction fn fn_id = vcat [
        matlabFunctionHeader fn,
        matlabHelp fn,
        -- TODO: move Matlab generation code to separate file Codegen.Matlab
        nest 4 $ C.statement $ matlabFunctionCall (text "nemo_mex") mexReturn mexArgs,
        text "end"
    ]
    where
        mexReturn = matlabFunctionReturn $ fn_output fn
        arg0 = text $ printf "uint32(%u)" fn_id
        userArgs = map (text . matlabInput . arg) $ fn_inputs fn
        mexArgs = arg0 : userArgs



-- TODO: tidy!
{- Some functions in the Matlab API are vectorised. The normal form of the
 - function in the C++ interface should have all scalar arguments. The
 - generated Matlab function accepts vector input (with vectors of the same
 - length) and internally calls the scalar version -}
{-
vectorizedMemberFunction :: Handle -> ApiFunction -> Int -> IO ()
vectorizedMemberFunction mfile fn idx = do
    putStrLn $ "Generating " ++ (fn_name fn)
    hPutStr mfile ( )
    where
        fname = fn_name fn
        fdef :: Doc
        fdef =
            -- void return and fixed function header
-}


commentBlock :: Doc -> Doc
commentBlock doc = vcat $ map (commentLine . text) $ lines $ render doc

commentLine :: Doc -> Doc
commentLine = (<+>) $ char '%'

description :: Doc -> Maybe Doc -> Doc
-- TODO: deal with long lines here
description term defn = term <+> (maybe empty ((char '-') <+>) defn)
-- description term defn = hang term 4 $ maybe empty ((char '-') <+>) defn


matlabArgDescription :: (Named a, Described a) => a -> Doc
matlabArgDescription x = h $$ nest indent t
    where
        n = name x
        indent = 10
        nspace = max 1 $ indent - length n - 2
        h = hcat $ [text n] ++ (replicate nspace space) ++ [char '-']
        t = maybe empty (fsep . map text . words) $ describe x



matlabHelp :: ApiFunction -> Doc
matlabHelp fn = commentBlock $ vcat $ [
            header, space,
            synopsis, space,
            inputs,
            outputs,
            descr,
            vectorization
        ]
    where
        fnname = matlabFunctionName fn

        header =  description (text fnname) (Just $ text $ fn_brief fn)

        synopsisCall = matlabFunctionCall (text fnname)
                (matlabFunctionReturn $ fn_output fn) $ map (text . name) $ fn_inputs fn

        synopsis = ($+$) (text "Synopsis:") $ nest 2 synopsisCall

        optlist :: (Named a, Described a) => String -> [a] -> Doc
        optlist title xs =
            if null xs
                then empty
                else hang (text title <> char ':') 2 $ vcat $ (map matlabArgDescription xs) ++ [space]

        inputs = optlist "Inputs" $ fn_inputs fn
        outputs = optlist "Outputs" $ fn_output fn

        descr = maybe empty reflow $ fn_descr fn

        reflow = fsep . map text . words

        vectorization =
            if fn_vectorized fn
                then char ' ' $+$ (reflow $ vectorization_inputs ++ vectorization_outputs)
                else empty

        vectorization_inputs = "The input arguments can be a mix of scalars and vectors as long as all vectors have the same length. Scalar arguments are replicated the appropriate number of times. "
        vectorization_outputs =
            if not ( null (fn_output fn))
                then "If all input arguments are scalar, the output is scalar. Otherwise the output has the same length as the vector input arguments. "
                else ""


-- TODO: make use of this in matlabHelp
synopsis :: ApiModule -> ApiFunction -> Doc
synopsis mdl fn = matlabFunctionCall fnname (matlabFunctionReturn $ fn_output fn) $ map (text . name) $ fn_inputs fn
    where
        obj = text $ mdl_sname mdl
        method = text $ name fn
        fnname = memberCall obj method


ctorSynopsis :: ApiModule -> Doc
ctorSynopsis mdl = Common.functionCall Common.None cname (Just sname) 60 args
    where
        sname = text $ mdl_sname mdl
        cname = text $ "nemo" ++ mdl_name mdl
        args  = map (text . mdl_sname) $ constructorArgs $ mdl_ctor mdl


memberCall :: Doc -> Doc -> Doc
memberCall obj method = obj <> char '.' <> method


uncamel [] = []
uncamel (x:xs)
    | isUpper x = '_' : toLower x : uncamel xs
    | otherwise = x : uncamel xs



mexFunctionName :: ApiFunction -> String
mexFunctionName = fn_name


cFunctionName :: ApiFunction -> String
cFunctionName = prefix "nemo_" . uncamel . fn_name
    where
        -- TODO: move to Codegen
        prefix pf str = pf ++ str


-- | All MEX functions have the same header
mexFunctionDefinition :: ApiFunction -> Doc -> Doc
mexFunctionDefinition fn body = functionDefinition ret name args body
    where
        ret = text "void"
        name = text $ fn_name fn
        args = map text ["int nlhs", "mxArray* plhs[]", "int nrhs", "const mxArray* prhs[]"]


-- TODO: add function mapping type to string
functionDefinition :: Doc -> Doc -> [Doc] -> Doc -> Doc
functionDefinition ret name args body = vcat [ret, proto, lbrace, nest 4 body, rbrace, nl, nl, nl]
    where
        -- TODO: do this like in arglist, with possible hanging
        proto = (<>) name $ parens $ hcat $ punctuate (text ", ") args
        nl = text ""


functionCall
    :: Doc          -- text to indicate multiline continuation
    -> Doc          -- function name
    -> Maybe Doc    -- output assignment
    -> [Doc]        -- arguments
    -> Doc
functionCall cont name output inputs = assign <> call
    where
        assign = maybe empty (<+> (text "= ")) output
        call = if length args_oneline > 60
                then vcat $ [name <> lparen <> cont] ++ args_multiline ++ [rparen]
                else (<>) name $ parens $ text args_oneline
        -- TODO: separate out multiline functionality
        args_oneline = render $ hcat $ punctuate (text ", ") inputs
        args_multiline = map (nest 8 . (<> cont)) $ punctuate (char ',') inputs

matlabFunctionCall = functionCall (text "...")
cFunctionCall = functionCall space


matlabFunctionHeader :: ApiFunction -> Doc
matlabFunctionHeader fn = (text "function") <+> call
    where
        call = matlabFunctionCall fname (matlabFunctionReturn $ fn_output fn) inputs
        fname = text $ matlabFunctionName fn
        inputs = map (text . name) $ fn_inputs fn


{- Output arguments are assigned to separate output variables -}
matlabFunctionReturn :: [OutputType] -> Maybe Doc
matlabFunctionReturn []  = Nothing
matlabFunctionReturn [x] = Just $ text $ name x
matlabFunctionReturn xs  = Just $ Common.arglistWith brackets $ map (text . name) xs


declareInit :: Doc -> Doc -> Doc -> Doc
declareInit t lhs rhs = t <+> lhs <+> (char '=') <+> rhs


{-| Return code for function which is found in either network or simulation module -}
mexScalarConstructibleFunction :: ApiFunction -> Doc
mexScalarConstructibleFunction fn = mexFunctionDefinition fn body
    where
        body = vcat $ [
                -- NOTE: input and output checks may be redundant
                C.statement $ cFunctionCall (text "checkInputCount") Nothing [text "nrhs", int (length $ fn_inputs fn)],
                C.statement $ cFunctionCall (text "checkOutputCount") Nothing [text "nlhs", int (length $ fn_output fn)],
                mexDeclareInputVariables 1 $ fn_inputs fn,
                mexDeclareOutputVariables $ fn_output fn,
                C.conditional (text "isSimulating()") (call "Simulation" "_s") (call "Network" "_n"),
                mexReturnOutputVariables $ fn_output fn
            ]
        call mdl_name suffix = C.statement $ cFunctionCall (text "checkNemoStatus") Nothing $
            [cFunctionCall (text $ (cFunctionName fn) ++ suffix) Nothing (ptr mdl_name : callArgs)]
        ptr mdl_name = cFunctionCall (hcat $ map text ["get", mdl_name]) Nothing []
        -- expand outputArgs so that they contain both vector and length
        -- make sure they are reference types as well
        callArgs = inputArgs ++ outputArgs
        inputArgs = zipWith mexInput [1..] $ fn_inputs fn
        outputArgs = map mexOutput $ fn_output fn



{-| Return vectorized code for function which is found in either network or simulation module -}
mexVectorConstructibleFunction :: ApiFunction -> Doc
mexVectorConstructibleFunction fn = mexFunctionDefinition fn body
    where
        body = vcat $ [
                -- NOTE: input and output checks may be redundant
                {- In the vector form, all inputs should have the same format.
                 - A pre-defined function can verify this. -}
                C.statement $ text "static unsigned arglen" <> (brackets $ int inputCount),
                C.statement $ cFunctionCall (text "vectorDimension") (Just $ text "size_t elems") [int inputCount, text "prhs + 1", text "arglen"],
                C.statement $ cFunctionCall (text "checkInputCount") Nothing [text "nrhs", int inputCount],
                C.statement $ cFunctionCall (text "checkOutputCount") Nothing [text "nlhs", int outputCount],
                mexDeclareInputVariables 1 $ fn_inputs fn,
                mexAllocateVectorOutputs $ fn_output fn,
                C.conditional (text "isSimulating()") (call "Simulation" "_s") (call "Network" "_n")
            ]

        -- Conditional with loop inside each branch
        call mdl_name suffix = vcat $ [
                C.statement $ cFunctionCall (getHandle mdl_name) (Just (handleType mdl_name <+> text "hdl")) [],
                C.forLoop indexVar "0" "elems" $ loopBody suffix
            ]

        -- Loop body iterates over inputs/outputs
        loopBody suffix = vcat $ [
                -- use a temporary scalar for the output
                mexDeclareOutputVariables $ fn_output fn,
                C.statement $ cFunctionCall (text "checkNemoStatus") Nothing $
                    [cFunctionCall (text (cFunctionName fn) <> text suffix) Nothing (text "hdl" : callArgs)],
                -- then convert to Matlab format and return
                mexVectorizedReturn indexVar $ fn_output fn
            ]

        getHandle mdl_name = (text "get") <> (text mdl_name)
        callArgs = inputArgs ++ outputArgs
        inputCount = length $ fn_inputs fn
        inputArgs = zipWith (mexVectorInput indexVar) [1..] $ fn_inputs fn
        outputCount = length $ fn_output fn
        outputArgs = map mexOutput $ fn_output fn
        indexVar = "i"




mexScalarFunction :: String -> ApiFunction -> Doc
mexScalarFunction mdl_name fn = mexFunctionDefinition fn body
    where
        body = vcat $ [
                -- NOTE: input and output checks may be redundant
                C.statement $ cFunctionCall (text "checkInputCount") Nothing [text "nrhs", int (length $ fn_inputs fn)],
                C.statement $ cFunctionCall (text "checkOutputCount") Nothing [text "nlhs", int (length $ fn_output fn)],
                mexDeclareInputVariables 1 $ fn_inputs fn,
                mexDeclareOutputVariables $ fn_output fn,
                C.statement $ cFunctionCall (text "checkNemoStatus") Nothing $
                        [cFunctionCall (text $ cFunctionName fn) Nothing (ptr : callArgs)],
                mexFreeTemporaries $ fn_inputs fn,
                mexReturnOutputVariables $ fn_output fn
            ]
        ptr = cFunctionCall getHandle Nothing []
        getHandle = (text "get") <> (text mdl_name)
        -- TODO: here!!!
        -- expand outputArgs so that they contain both vector and length
        -- make sure they are reference types as well
        callArgs = inputArgs ++ outputArgs
        inputArgs = zipWith mexInput [1..] $ fn_inputs fn
        outputArgs = map mexOutput $ fn_output fn


mexVectorFunction :: String -> ApiFunction -> Doc
mexVectorFunction mname fn = mexFunctionDefinition fn body
    where
        body = vcat $ [
                {- In the vector form, all inputs should have the same format.
                 - A pre-defined function can verify this. -}
                C.statement $ text "static unsigned arglen" <> (brackets $ int inputCount),
                C.statement $ cFunctionCall (text "vectorDimension") (Just $ text "size_t elems") [int inputCount, text "prhs + 1", text "arglen"],
                C.statement $ cFunctionCall (text "checkInputCount") Nothing [text "nrhs", int inputCount],
                C.statement $ cFunctionCall (text "checkOutputCount") Nothing [text "nlhs", int outputCount],
                mexDeclareInputVariables 1 $ fn_inputs fn,
                mexAllocateVectorOutputs $ fn_output fn,
                -- get the handle argument only once to reduce overhead, esp. for error checking
                C.statement $ cFunctionCall getHandle (Just (handleType mname <+> text "hdl")) [],
                C.forLoop indexVar "0" "elems" loopBody
                -- mexReturnOutputVariables $ fn_output fn
            ]
        -- ptr = cFunctionCall getHandle Nothing [text "prhs", int 1]
        getHandle = (text "get") <> (text mname)
        callArgs = inputArgs ++ outputArgs
        inputCount = length $ fn_inputs fn
        inputArgs = zipWith (mexVectorInput indexVar) [1..] $ fn_inputs fn
        outputCount = length $ fn_output fn
        outputArgs = map mexOutput $ fn_output fn
        indexVar = "i"
        loopBody = vcat $ [
                -- use a temporary scalar for the output
                mexDeclareOutputVariables $ fn_output fn,
                C.statement $ cFunctionCall (text "checkNemoStatus") Nothing $ [cFunctionCall (text $ cFunctionName fn) Nothing (text "hdl" : callArgs)],
                -- then convert to Matlab format and return
                mexVectorizedReturn indexVar $ fn_output fn
            ]

{- For vectors we need to pass a pointer and a length separately, so we need to
 - create the vector before the call. This also performs the cast from Matlab
 - types if required. In the MEX layer we don't deal with optional arguments.
 -}
mexDeclareInputVariables :: Int -> [Input] -> Doc
mexDeclareInputVariables _ [] = empty
mexDeclareInputVariables firstarg xs = vcat $ zipWith singleDecl [firstarg..] $ map arg xs
    where
        singleDecl :: Int -> ApiArg -> Doc
        singleDecl argno x =
            if scalar x
                then case baseType x of
                    ApiString -> mexDeclareInputString argno x
                    _         -> empty
                else mexDeclareInputVector argno x


mexDeclareInputString argno x = C.statement $ cFunctionCall call decl [text $ printf "prhs[%u]" argno]
    where
        call = text "mxArrayToString"
        decl = Just $ text "char*" <+> text (name x)

mexDeclareInputVector argno x = C.statement $ cFunctionCall (call x) (decl x) [text $ printf "prhs[%u]" argno]
    where
        mtype = mexType . baseType . arg_type
        ntype = cppType . baseType . arg_type
        call x = text $ printf "vector<%s, %s>" (ntype x) (mtype x)
        decl x = Just $ text $ printf "std::vector<%s> %s" (ntype x) $ name x


{- input strings need to manually managed -}
mexFreeTemporaries :: [Input] -> Doc
mexFreeTemporaries xs = vcat $ map (go . arg) xs
    where
        go :: ApiArg -> Doc
        go x =
            case baseType x of
                ApiString -> C.statement $ cFunctionCall (text "mxFree") Nothing [text $ name x]
                _         -> empty



{- Since we use the C interface, outputs are returned via pointers. Currently
 - we only auto-generate scalar returns -}
mexDeclareOutputVariables :: [OutputType] -> Doc
mexDeclareOutputVariables [] = empty
mexDeclareOutputVariables xs = vcat $ map singleDecl xs
    where
        singleDecl :: OutputType -> Doc
        singleDecl x = vcat [mainDecl x , vectorLen x]

        mainDecl x = (typeDecl $ arg_type x) <+> (text $ name x) <> char ';'

        vectorLen x = if explicitLength (fullType x)
                        then (text $ "size_t " ++ name x ++ "_len;")
                        else empty

        typeDecl :: Type -> Doc
        typeDecl (Scalar t) = text $ cppType t
        -- For vectors we need both an output pointer and the length
        typeDecl (Vector t _) = pointer t
        --  typeDecl (Vector t) = error "vector output not supported"


mexReturnOutputVariables :: [OutputType] -> Doc
mexReturnOutputVariables [] = empty
mexReturnOutputVariables xs = vcat $ zipWith go [0..] xs
    where
        go :: Int -> OutputType -> Doc
        -- go argno arg =
        --    let t = baseType $ arg_type arg
        --    in text $ printf "returnScalar<%s, %s>(plhs, %u, %s);" (cppType t) (mexType t) argno (name arg)
        -- go (Vector t) = printf "returnVector<%s, %s>(plhs, %s)" (cppType t) (mexType t) str
        go argno arg = text $
            if scalar arg
                then printf "returnScalar<%s, %s>(plhs, %u, %s);" (cppType t) (mexType t) argno n
                else printf "returnVector<%s, %s>(plhs, %u, %s, %s);" (cppType t) (mexType t) argno n len
            where
                t = baseType $ arg_type arg
                n = name arg

                len =
                    case fullType arg of
                        Scalar _ -> error "unexpected scalar"
                        Vector _ vlen ->
                            case vlen of
                                ExplicitLength -> n ++ "_len"
                                (ImplicitLength x) -> name x ++ ".size()"



mexVectorizedReturn :: String -> [OutputType] -> Doc
mexVectorizedReturn _ [] = empty
mexVectorizedReturn idx xs = vcat $ zipWith go [0..] xs
    where
        go :: Int -> OutputType -> Doc
        go argno arg = text $
            if scalar arg
                then printf "returnScalarAt<%s, %s>(plhs, %u, %s, %s);" (cppType t) (mexType t) argno idx (name arg)
                else error "Vectorized function with non-scalar output in underlying API function"
            where
                t = baseType $ arg_type arg
                n = name arg



mexAllocateVectorOutputs :: [OutputType] -> Doc
mexAllocateVectorOutputs [] = empty
mexAllocateVectorOutputs xs = vcat $ zipWith go [0..] xs
    where
        go :: Int -> OutputType -> Doc
        go argno arg =
            if scalar arg
                then C.statement $ cFunctionCall (text "allocateOutputVector" <> Common.angleBrackets (text $ mexType t)) Nothing [text "plhs", int argno, text "elems"]
                else error "Vectorized function with non-scalar output in underlying API function"
            where
                t = baseType $ arg_type arg


{- We use a single MEX file with a single exposed function. Dispatch to the
 - correct function is done via the first input argument. mexFuncionTable
 - prints the table of function pointers. -}
mexFunctionTable :: [String] -> Doc
mexFunctionTable fnames = vcat [
        text "typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);",
        text ("#define FN_COUNT " ++ show (length fnames)),
        text "fn_ptr fn_arr[FN_COUNT] = {",
        (vcat $ map (nest 4) $ punctuate (char ',') $ map text fnames),
        text "};\n"
    ]


{- In the MEX layer we must use types that Matlab can work with -}
mexType :: BaseType -> String
mexType ApiFloat = "double"
mexType ApiUInt = "uint32_t"
mexType ApiInt = "int32_t"
mexType ApiUInt64 = "uint64_t"
mexType ApiULong = "uint64_t"
mexType ApiBool = "uint8_t"
mexType ApiString = "char*"


{- In the Matlab layer we need to translate from whatever random type the input
 - is (probably double) to a sensible format -}
cppType :: BaseType -> String
cppType ApiFloat = "float"
cppType ApiUInt = "unsigned"
cppType ApiUInt64 = "uint64_t"
cppType ApiInt = "int"
cppType ApiULong = "unsigned long"
cppType ApiBool = "unsigned char"
cppType ApiString = "const char*"


pointer :: BaseType -> Doc
pointer baseType = (text $ cppType baseType) <> char '*'


mexInput :: Int -> Input -> Doc
mexInput argno a = text $
    if scalar a
        then case baseType a of
            ApiString -> name a
            _         -> printf "scalar<%s,%s>(prhs[%d])" (cppType t) (mexType t) argno
        else if explicitLength (fullType a)
            then printf "&%s[0], %s.size()" n n
            -- TODO: add length check for implicit inputs
            else printf "&%s[0]" n
    where
        t = baseType a
        n = name a


{- Input argument for vectorized function. This is vectorized in the MEX layer,
 - i.e. the input is a Matlab matrix/vector and we access the C API for each
 - element. -}
mexVectorInput :: String -> Int -> Input -> Doc
mexVectorInput idx argno a = text $
    if scalar a
        then printf "scalarAt<%s,%s>(prhs[%d], %s, arglen[%d])" (cppType t) (mexType t) argno idx (argno-1)
        else error "mexVectorInput called with input which is already a vector"
    where
        t = baseType a
        n = name a


mexOutput :: OutputType -> Doc
-- mexOutput = addressOf . text . name
mexOutput arg = text $
    if scalar arg
        then printf "&%s" n
        else if explicitLength (fullType arg)
            then printf "&%s, &%s_len" n n
            else printf "&%s" n
    where
        n = name arg


matlabType :: BaseType -> String
-- TODO: do the conversion on the matlab side instead
matlabType ApiFloat = "double" -- these are converted on the C side
matlabType ApiString = error "strings are not cast"
matlabType ApiUInt = "uint32"
matlabType ApiUInt64 = "uint64"
matlabType ApiInt = "int32"
matlabType ApiULong = "uint64"
matlabType ApiBool = "uint8"


{-- | convert module name to C API handle typename -}
handleType :: String -> Doc
handleType mdl = text $ "nemo_" ++ (map toLower mdl) ++ "_t"


matlabInput :: ApiArg -> String
matlabInput arg =
    case t of
        ApiString -> name arg
        _         -> printf "%s(%s)" (matlabType t) (name arg)
    where
        t = baseType $ arg_type arg

