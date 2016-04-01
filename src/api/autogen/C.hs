module C (
    statement,
    synopsis,
    ctorSynopsis,
    constant,
    pointerTo,
    forLoop,
    conditional,
    addressOf,
    comment,
    commentLine)
where

import Text.PrettyPrint
import Text.Printf
import Data.Char (toLower)

import API
import Common


ctorSynopsis :: ApiModule -> Doc
ctorSynopsis mdl = (go $ mdl_ctor mdl) $+$ char ' ' $+$ dtorSynopsis mdl
    where
        go :: Constructor -> Doc
        -- no need for pointer here, the return is already a pointer type
        go (Constructor args) = moduleType mdl $+$
            (functionDefinition (Compact empty) ctorName Nothing 40 $
            map moduleArg args)
        go (Factory args) = go (Constructor args)

        ctorName = scopedName $ underscoredName ["new", map toLower (name mdl)] 


dtorSynopsis :: ApiModule -> Doc
dtorSynopsis mdl = functionDefinition None dtorName Nothing 40 $ [moduleArg mdl]
    where
        dtorName = scopedName $ underscoredName ["delete", map toLower (name mdl)]



synopsis :: ApiModule -> ApiFunction -> Doc
synopsis mdl fn =
    if C `elem` fn_noauto fn || length os > 1
        then text "cannot auto-generate synopsis"
        else text "nemo_status_t" $+$ functionDefinition (Compact empty) fname Nothing 56 args
    where
        os = fn_output fn
        fname = scopedName $ text $ camelToUnderscore $ name fn
        args = mdl_arg : iargs ++ oargs
        mdl_arg = moduleArg mdl
        iargs = map (argument input) $ fn_inputs fn
        oargs = map (argument output) $ fn_output fn
        -- iargs = map inputArgument $ fn_inputs fn
        -- oargs = map outputArgument $ fn_output fn



{- For lack of namespaces, we prefix functions, types etc -}
scopedName :: Doc -> Doc
scopedName doc = text "nemo_" <> doc


{- Type names are postfixed with _t -}
typePostfixed :: Doc -> Doc
typePostfixed doc = doc <> text "_t"


{- Each module/class has a name of the form nemo_xxx_t -}
moduleType :: ApiModule -> Doc
moduleType = scopedName . typePostfixed . text . map toLower . mdl_name


{- For each function in the C API, the first argument is a pointer to the
 - underlying C++ object. -}
moduleArg :: ApiModule -> Doc
moduleArg mdl = moduleType mdl <+> text (mdl_sname mdl)


-- type: name pair
type FunctionArgument = (Doc, Doc)


functionArgument :: (Typed a, Named a) => a -> FunctionArgument
functionArgument x = (typeName (baseType x), text $ name x)


fa_array :: FunctionArgument -> FunctionArgument
fa_array (t, n) = (t, n <> text "[]")


fa_pointerTo :: FunctionArgument -> FunctionArgument
fa_pointerTo (t, n) = (t <> char '*', n)


fa_doc :: FunctionArgument -> Doc
fa_doc (t, n) = t <+> n


argument :: (Dimensional a, Named a, Typed a) => (FunctionArgument -> FunctionArgument) -> a -> Doc
argument f arg
        | scalar arg = fa_doc $ f $ functionArgument arg
        | vector arg = vectorArgument f arg


{- A vector argument *may* have an associated length (of type size_t) -}
vectorArgument :: (Typed a, Named a) => (FunctionArgument -> FunctionArgument) -> a -> Doc
vectorArgument f arg
    | explicitLength t = listPunctuate $ map (fa_doc . f) $ [vecArg, lengthArg]
    | otherwise        = fa_doc $ f vecArg
    where
        t = fullType arg
        vecArg = fa_array $ functionArgument arg
        lengthArg = (text "size_t", text (name arg) <> text "_len")


input :: FunctionArgument -> FunctionArgument
input = id

output :: FunctionArgument -> FunctionArgument
output = fa_pointerTo



{-
input :: (Named a, Typed a) => a -> FunctionArgument
input x = (typeName (baseType x), text $ name x)


output :: (Named a, Typed a) => a -> FunctionArgument
output x = fa_pointerTo . input
-}



-- scalarArgument :: ApiArg -> Doc
-- scalarArgument arg = (typeName $ baseType arg) <+> (text $ name arg)


{- A vector argument /may/ have an associated length (of type size_t) -}
{-
vectorArgument :: (ApiArg -> Doc) -> ApiArg -> [Doc]
vectorArgument f arg@(ApiArg _ _ t)
    | explicitLength t = [mainArg arg, lengthArg arg]
    | otherwise        = [mainArg arg]
    where
        mainArg = array . f
        lengthArg = 
        lengthArg arg = text "size_t" <+> text (name arg) <> text "_len"
        -- mainArg = array . scalarArgument
        -- lengthArg arg = text "size_t" <+> text (name arg) <> text "_len"


inputArgument :: Input -> Doc
inputArgument a
        | scalar a = scalarArgument $ arg a
        | vector a = listPunctuate $ vectorArgument input $ arg a


outputArgument :: ApiArg -> Doc
outputArgument a
        | scalar a = pointerTo $ scalarArgument a
        | vector a = listPunctuate $ vectorArgument output a


input :: ApiArg -> Doc
input x = (typeName $ baseType x) <+> (text $ name x)


output :: ApiArg -> Doc
output x = (pointerTo $ typeName $ baseType x) <+> (text $ name x)
-}
{-
argument :: (Dimensional a, Named a, Typed a) => (a -> Doc) -> a -> Doc
argument f arg
        | scalar arg = f arg
        | vector arg = array $ f arg


input :: (Named a, Typed a) => a -> Doc
input x = (typeName $ baseType x) <+> (text $ name x)


output :: (Named a, Typed a) => a -> Doc
output x = (pointerTo $ typeName $ baseType x) <+> (text $ name x)
-}


constant :: Doc -> Doc
constant c = text "const" <+> c


pointerTo :: Doc -> Doc
pointerTo x = x <> char '*'


array :: Doc -> Doc
array x = x <> text "[]"


typeName :: BaseType -> Doc
typeName = text . go
    where
        go ApiFloat = "float"
        go ApiUInt = "unsigned"
        go ApiUInt64 = "uint64_t"
        go ApiInt = "int"
        go ApiULong = "unsigned long"
        go ApiBool = "unsigned char"
        go ApiString = "const char*"


{- Idiomatic for loop -}
forLoop
    :: String  -- ^ indexing variable
    -> String  -- ^ start
    -> String  -- ^ end
    -> Doc     -- ^ body of loop
    -> Doc
forLoop idx start end body = vcat [header, nest 4 body, char '}']
    where
        header = text $ printf "for(size_t %s=%s; %s<%s; ++%s){" idx start idx end idx

conditional
    :: Doc    -- ^ conditional
    -> Doc    -- ^ if-clause
    -> Doc    -- ^ else-clause
    -> Doc
conditional cond ifclause elseclause = vcat [
        text "if" <> parens cond <> lbrace,
        nest 4 ifclause,
        rbrace <+> text "else" <+> lbrace,
        nest 4 elseclause,
        rbrace
    ]


addressOf :: Doc -> Doc
addressOf x = char '&' <> x


comment :: Doc -> Doc
comment x = text "/*" <+> x <+> text "*/"


commentLine :: Doc -> Doc
commentLine = (<+>) $ text "//"


statement :: Doc -> Doc
statement x = x <> char ';'

statements :: [Doc] -> Doc
statements = vcat . map statement
