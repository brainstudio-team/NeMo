module Cpp (synopsis, ctorSynopsis) where

import Text.PrettyPrint

import API
import Common

import C (constant, pointerTo)


ctorSynopsis :: ApiModule -> Doc
ctorSynopsis mdl = go $ mdl_ctor mdl
    where
        go :: Constructor -> Doc
        go (Constructor args) = def ctorName args
        go (Factory args) = factoryReturn $+$ def factoryName args

        def name args = functionDefinition (Compact empty) name Nothing width $ map ctor_arg args

        ctorName = memberFunction mdl $ text $ name mdl
        factoryName = text $ camelToUnderscore $ name mdl

        factoryReturn = pointerTo $ text $ name mdl

        ctor_arg arg = constant $ ref $ text $ name arg

        width = 60



synopsis :: ApiModule -> ApiFunction -> Doc
synopsis mdl fn =
    if CPP `elem` fn_noauto fn || length os > 1
        then text "cannot auto-generate synopsis"
        else output fn $+$ functionDefinition (Compact empty) fname Nothing 56 args
    where
        os = fn_output fn
        fname = memberFunction mdl $ text (name fn)
        args = map inputArg $ fn_inputs fn


memberFunction :: ApiModule -> Doc -> Doc
memberFunction mdl fn = text (mdl_name mdl) <> text "::" <> fn



output :: ApiFunction -> Doc
output fn = go $ fn_output fn
    where
        go [] = text "void"
        go [x] = singleOutput x
        -- TODO: throw an error here?
        go xs = text "unknown"


inputArg :: Input -> Doc
inputArg x = (go $ arg_type a) <+> (text $ name a)
    where
        -- TODO: distinguish between optional and required arguments
        a = arg x
        go (Scalar t) = scalarInput t
        go (Vector t _) = vectorInput t


singleOutput :: OutputType -> Doc
singleOutput x = go $ arg_type x
    where
        go (Scalar t) = scalarOutput t
        go (Vector t _) = vectorOutput t


scalarVal :: BaseType -> Doc
scalarVal = text . typeName


scalarOutput = scalarVal
scalarInput = scalarVal



vectorVal :: BaseType -> Doc
vectorVal t = text "vector" <> (angleBrackets $ text $ typeName t)


vectorOutput = constant . ref . vectorVal

vectorInput :: BaseType -> Doc
vectorInput = constant . ref . vectorVal



ref :: Doc -> Doc
-- ref c = c <> text "\\&"
ref c = c <> text "&"


-- TODO: share this with Matlab code
{- In the Matlab layer we need to translate from whatever random type the input
 - is (probably double) to a sensible format -}
typeName :: BaseType -> String
typeName ApiFloat = "float"
typeName ApiUInt = "unsigned"
typeName ApiUInt64 = "uint64_t"
typeName ApiInt = "int"
typeName ApiULong = "unsigned long"
typeName ApiBool = "unsigned char"
typeName ApiString = "std::string"
